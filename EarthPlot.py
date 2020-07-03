from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, Texture, TextureStage, DirectionalLight, VBase4, VBase2, AmbientLight, LineSegs, GeomPoints, NodePath, CompassEffect, MouseButton, SamplerState
from direct.gui.OnscreenText import OnscreenText
from PIL import Image
import threading
import numpy as np
from matplotlib import cm
import shapefile
from QPanda3D.Panda3DWorld import Panda3DWorld
from QPanda3D.QPanda3DWidget import QPanda3DWidget
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore    import Qt
from PyQt5.QtGui import QCursor
import sys

class QPanda3DWidget_(QPanda3DWidget):
    def __init__(self, *args,**kwargs):
        QPanda3DWidget.__init__(self, *args, **kwargs)
        self.button1_held = False
    def mousePressEvent(self,e):
        if e.button() == Qt.LeftButton:
            self.button1_held = True
            self.current_mouse_position = e.pos()
            print(e.pos()   )
            print('button pressed')
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            self.button1_held = False
            print('button released')

class EarthPlot(Panda3DWorld):
    def __init__(self,show_poles = False, highres_earth = False, d_light = True, d_light_HPR = (90,0,23.5), d_light_strength = (1,1,1,1), *args,**kwargs):
        Panda3DWorld.__init__(self,*args,**kwargs)
        self.canvas_width = 900
        self.canvas_height = 900
        self.sphere_size = 100
        self.time_elapsed = 0 #seconds
        self.earth_radius = 6378.137 #km
        self.plot_radius = self.earth_radius+5
        self.time_step = 10 #seconds
        self.groundstations = [] #list for current groundstation objects
        self.drag_id = ''
        self.x_mouse_position = 0
        self.y_mouse_position = 0
        self.qcursor = QCursor()
        self.props = WindowProperties()
        # self.props.setOrigin(100, 100)
        self.props.setSize(int(self.canvas_width), int(self.canvas_height))
        # self.openDefaultWindow(props=self.props)
        self.setBackgroundColor(0,0,0)

        #load sphere model
        self.earth_nodepath= loader.loadModel('./resources/sphere.egg')
        self.earth_nodepath.reparentTo(self.render)
        self.earth_nodepath.setScale(self.earth_radius/self.sphere_size)
        #enable shaders (gloss, ..)
        self.earth_nodepath.setShaderAuto()
        #initiate textures
        self.RGB_tex = Texture()
        self.gloss_tex = Texture()
        self.emis_tex = Texture()

        #loading images
        if highres_earth:
            img = Image.open("./resources/8081_earthmap10k.jpg")
            gloss_img =  Image.open("./resources/8081_earthspec10k.jpg")
        else:
            img = Image.open("./resources/8081_earthmap4k.jpg")
            gloss_img =  Image.open("./resources/8081_earthspec4k.jpg")
        img = np.flipud(np.array(img))
        gloss_img = np.flipud(np.array(gloss_img)) *0.7
        self.resolution = img.shape

        #setting RGB texture
        self.RGB_tex.setup2dTexture(self.resolution[1],self.resolution[0], Texture.T_unsigned_byte, Texture.F_rgb8)
        self.RGB_tex.setMagfilter(SamplerState.FT_linear)
        RGB_buff = img.astype(np.uint8).tobytes()
        self.RGB_tex.setRamImageAs(RGB_buff, 'RGB')
        self.earth_nodepath.setTexture(self.RGB_tex)

        #setting gloss/specularity texture
        gloss_ts = TextureStage('glossmap')
        gloss_ts.setMode(TextureStage.MGloss)
        self.gloss_tex.setup2dTexture(self.resolution[1],self.resolution[0], Texture.T_unsigned_byte, Texture.F_alpha)
        gloss_buff = gloss_img.astype(np.uint8).tobytes()
        self.gloss_tex.setRamImage(gloss_buff)
        self.earth_nodepath.setTexture(gloss_ts,self.gloss_tex)

        #lights
        #directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor(VBase4(d_light_strength[0], d_light_strength[1], d_light_strength[2], d_light_strength[3]))
        self.dlnp = render.attachNewNode(dlight)
        self.dlnp.setHpr(d_light_HPR[0], d_light_HPR[1], d_light_HPR[2])
        if d_light:
            render.setLight(self.dlnp)

        #amblight earth
        amblight_earth = AmbientLight('amblight_earth')
        if d_light:
            amblight_earth.setColor(VBase4(.5, .5, .5, 1))
        else:
            amblight_earth.setColor(VBase4(.9, .9, .9, 1))
        self.alnp = self.earth_nodepath.attachNewNode(amblight_earth)
        self.earth_nodepath.setLight(self.alnp)
        #amblight lines
        self.amblight_lines = AmbientLight('amblight_lines')
        self.amblight_lines.setColor(VBase4(1, 1, 1, 1))
        if show_poles:
            earth_poles = LineSegs()
            earth_poles.setThickness(4)
            earth_poles.setColor(1,0,0)
            earth_poles.moveTo(0,0,0)
            earth_poles.drawTo(0,0,-6600)
            earth_poles.moveTo(0,0,0)
            earth_poles.setColor(0,0,1)
            earth_poles.drawTo(0,0,6600)
            node = earth_poles.create()
            line_np = NodePath(node)
            line_np.reparentTo(self.render)
            alnp = line_np.attachNewNode(self.amblight_lines)
            line_np.setLight(alnp)

        #camera settings
        self.disableMouse()
        self.parentnode = render.attachNewNode('camparent')
        self.parentnode.reparentTo(self.earth_nodepath) # inherit transforms
        self.parentnode.setEffect(CompassEffect.make(render)) # NOT inherit rotation
        self.camera.reparentTo(self.parentnode)
        self.camera.setY(-self.earth_radius/15) # camera distance from model
        self.camera.lookAt(self.parentnode)
        self.heading = 0
        self.pitch = 0
        self.taskMgr.add(self.OrbitCameraTask, 'thirdPersonCameraTask')
        self.accept('wheel_up', lambda : self.set_camera_fov(self.camLens.getFov(),+0.5))
        self.accept('wheel_down', lambda : self.set_camera_fov(self.camLens.getFov(),-0.5))

    def set_camera_fov(self,fov, value):
        new_fov = fov[0] + value
        H = new_fov*np.pi/180
        x_size = self.win.getXSize()
        y_size = self.win.getYSize()
        inv_aspect = y_size/x_size
        V = 2*np.arctan(np.tan(H/2)* inv_aspect)*180/np.pi
        fov_lr = new_fov
        fov_ud = V
        if (fov_lr > 1) and (fov_ud > 1):
            if (fov_lr <= 120) and (fov_ud <= 120):
                self.camLens.setFov(VBase2(fov_lr, fov_ud))
        return

    def show(self):
        self.run()

    def plot_lines(self,xs,ys,zs, color = [0,0,0], linewidth = 2, shading = 'None'):
        '''
        takes 3 coordinates in a list and plots them as continuing line
        '''
        line = [(x,y,z) for x,y,z in zip(xs,ys,zs)]
        lines = LineSegs()
        lines.setColor(color[0], color[1], color[2])
        for point in line:
            lines.drawTo(point[0],point[1],point[2])
        lines.setThickness(linewidth)
        node = lines.create()
        line_np = NodePath(node)
        line_np.reparentTo(self.render)
        if shading == 'None':
            alnp = line_np.attachNewNode(self.amblight_lines)
            line_np.setLight(alnp)
        elif shading == 'ambient':
            line_np.setLight(self.alnp)
        elif shading == 'directional':
            line_np.setLight(self.dlnp)
        else:
            raise ValueError('shading must be either "None", "ambient" or "directional".')
        return line_np

    def plot_markers(self,xs,ys,zs, color = [0,0,0], markersize = 1):
        '''
        takes list of coordinates and plots a spherical marker on each one
        '''
        np_list = []
        points = [(x,y,z) for x,y,z in zip(xs,ys,zs)]
        for point in points:
            np = loader.loadModel('./resources/sphere.egg')
            np.reparentTo(self.render)
            alnp = np.attachNewNode(self.amblight_lines)
            np.setLight(alnp)
            np.setColor(color[0], color[1], color[2])
            np.setPos(point[0], point[1], point[2])
            np.setScale(markersize)
            np_list.append(np)

        return np_list

    def calc_orbit(self,a,e,Omega,i,omega, resolution = 100):
        '''calculates orbit 3x1 radius vectors from kepler elements'''
        nu = np.linspace(0,2*np.pi,resolution)
        if e <1:
            p = a * (1-(e**2))
            r = p/(1+e*np.cos(nu))
            r = np.array([np.multiply(r,np.cos(nu)) , np.multiply(r,np.sin(nu)), np.zeros(len(nu))])
            r = np.matmul(self.rot_z(omega),r)
            r = np.matmul(self.rot_x(i),r)
            r = np.matmul(self.rot_z(Omega),r)
        elif e >=1:
            raise ValueError('eccentricity must be smaller than 1, hyperbolic and parabolic orbits are not supported')
        return r

    @staticmethod
    def rot_x(phi):
        '''returns rotational matrix around x, phi in rad'''
        return np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
    @staticmethod
    def rot_z(rho):
        '''returns rotational matrix around z, rho in rad'''
        return np.array([[np.cos(rho),-np.sin(rho),0],[np.sin(rho),np.cos(rho),0],[0,0,1]])

    def plot_orbit(self,a,e,Omega,i,omega, resolution = 100, **args):
        r = self.calc_orbit(a,e,Omega,i,omega, resolution)
        xs = r[0]
        ys = r[1]
        zs = r[2]
        np = self.plot_lines(xs,ys,zs, **args)
        return np

    def plot_surface_markers(self, lons, lats, **args):
        xs, ys, zs = [], [], []
        np_list = []

        for lon,lat in zip(lons,lats):
            x, y, z = self.calculate_cartesian(lon,lat)
            xs.append(x), ys.append(y), zs.append(z)
        np_list = self.plot_markers(xs,ys,zs, **args)
        return np_list

    def plot_surface_lines(self, lons, lats, **args):
        xs, ys, zs = [], [], []
        np_list = []
        for lon,lat in zip(lons,lats):
            x, y, z = self.calculate_cartesian(lon,lat)
            xs.append(x), ys.append(y), zs.append(z)
        np_list = self.plot_lines(xs,ys,zs, **args)
        return np_list

    def calculate_cartesian(self, lon, lat):
        lon, lat = np.pi*lon/180, np.pi*lat/180
        x = self.plot_radius * np.cos(lat) * np.cos(lon)
        y = self.plot_radius * np.cos(lat) * np.sin(lon)
        z = self.plot_radius * np.sin(lat)
        return x,y,z

    def plot_greatcircle(self, lon1,lat1, lon2,lat2, resolution = 500, **args):
        lons,lats = self.greatcircle_fun(lon1,lat1, lon2,lat2, resolution = resolution)
        nodepath = self.plot_surface_lines(lons,lats, **args)
        return nodepath

    def greatcircle_fun(self, lon1,lat1, lon2,lat2, resolution = 500):
        lons = np.linspace(0,2*np.pi,resolution)
        lat1, lon1 = np.pi*lat1/180, np.pi*lon1/180
        lat2, lon2 = np.pi*lat2/180, np.pi*lon2/180
        lats =  np.arctan( np.tan(lat1)*( np.sin(lons - lon2) / np.sin(lon1 - lon2) ) - np.tan(lat2)* ( np.sin(lons-lon1) / np.sin(lon1 - lon2)) )
        lons = 180*lons/np.pi
        lats = 180*lats/np.pi
        return lons, lats

    def plot_geodetic(self, lon1,lat1, lon2,lat2, resolution = 50,**args):
        lat1, lon1 = np.pi*lat1/180, np.pi*lon1/180
        lat2, lon2 = np.pi*lat2/180, np.pi*lon2/180
        lamb_12 = lon2 - lat1
        alpha_1 = np.arctan2 ( (np.cos(lat2)*np.sin(lamb_12)) ,  (np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lamb_12)) )
        alpha_2 = np.arctan2 ( (np.cos(lat1)*np.sin(lamb_12)) , -(np.cos(lat2)*np.sin(lat1) + np.sin(lat2)*np.cos(lat1)*np.cos(lamb_12)) )
        sigma_12 = np.arctan ( np.sqrt((np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lamb_12))**2 + (np.cos(lat2)*np.sin(lamb_12))**2  ) / (np.sin(lat1) * np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lamb_12)) )
        alpha_0 = np.arctan( np.sin(alpha_1)*np.cos(lat1) / np.sqrt(np.cos(alpha_1)**2 + np.sin(alpha_1)**2 * np.sin(lat1)**2))
        if lat1 == 0 or np.isclose(alpha_1,np.pi/2):# check for alpha_1 == pi/2 with epsilon
            sigma_01 = 0
        else:
            sigma_01 = np.arctan2( np.tan(lat1) , np.cos(alpha_1))

        sigma_02 = sigma_01 + sigma_12
        lamb_01 = np.arctan( np.sin(alpha_0)*np.sin(sigma_01)/np.cos(sigma_01))
        lamb_0 = lon1 - lamb_01
        sigmas = np.linspace(sigma_01, sigma_02, resolution)
        lats = np.arctan( np.cos(alpha_0)*np.sin(sigmas) / np.sqrt(np.cos(sigmas)**2 + np.sin(alpha_0)**2 * np.sin(sigmas)**2))
        lons = np.arctan2( np.sin(alpha_0)*np.sin(sigmas) , np.cos(sigmas)) + lamb_0
        lats, lons = 180*lats/np.pi, 180*lons/np.pi
        nodepath = self.plot_surface_lines(lons,lats, **args)
        return nodepath

    def plot_heatmap(self, lons, lats, n_bins = 200, mpl_cm = 'jet', mpl_cm_steps = 12, alpha = .9, **args):
        extent = [-180,180,-90,90]
        extent2 =  [[-180,180],[-90,90]]
        lons = [(lon+180)%360-180 for lon in lons]
        H, xedges, yedges = np.histogram2d(lons,lats, bins = n_bins, range = extent2)
        H = H-np.amin(H)
        H = H/np.amax(H)
        H = np.transpose(H)
        colormap = cm.get_cmap(mpl_cm, mpl_cm_steps)
        shape = H.shape
        image = colormap(H)
        #set alpha to heatmap data:
        image[:,:,3] = H*alpha
        image = image*255
        H = H*255
        tex,ts = self.imshow(image, **args)
        return tex,ts

    def imshow(self, image, texture_filter = 'None'):
        tex = Texture()
        ts = TextureStage('heatmap')
        ts.setMode(TextureStage.MDecal)
        resolution = image.shape
        if resolution[2] == 4:
            tex_type = Texture.F_rgba
        else:
            tex_type = Texture.F_rgb8
        if not texture_filter == 'None':
            if texture_filter == 'nearest':
                texture_filter = SamplerState.FT_nearest
            elif texture_filter == 'linear':
                texture_filter = SamplerState.FT_linear
            else:
                raise ValueError('texture_filter needs to be either "nearest", "linear" or "None".')
            tex.setMagfilter(texture_filter)
        tex.setup2dTexture(resolution[1],resolution[0], Texture.T_unsigned_byte, tex_type)

        RGBA_buff = image.astype(np.uint8).tobytes()
        tex.setRamImageAs(RGBA_buff, 'RGBA')
        self.earth_nodepath.setTexture(ts, tex)
        return ts, tex

    def OrbitCameraTask(self,task):
        delta_factor = .01 * self.camLens.getFov()[1]
        # md = self.win.getPointer(0)
        mouse = self.qcursor.pos()
        x = mouse.x()
        y = mouse.y()
        if self.parent.button1_held:
            self.parent.setCursor(Qt.BlankCursor)
            self.qcursor.setPos(self.x_mouse_position, self.y_mouse_position)
            self.heading = self.heading - (x - self.x_mouse_position) * delta_factor
            self.pitch = self.pitch - (y - self.y_mouse_position) * delta_factor
            if self.pitch > 90:
                self.pitch = 90
            elif self.pitch <-90:
                self.pitch = -90
            self.parentnode.setHpr(self.heading, self.pitch,0)
        else:
            self.parent.setCursor(Qt.ArrowCursor)
            self.x_mouse_position = x
            self.y_mouse_position = y
        return task.cont

    def plot_shape(self,shape, **args):
        points = shape.points
        lons = []
        lats = []
        for i in range(0,len(points)):
            p1 = points[i]
            p2 = points[(i+1)%len(points)]
            lon = [p1[0], p2[0]]
            lat = [p1[1], p2[1]]
            lons = np.concatenate((lons,lon))
            lats = np.concatenate((lats,lat))

        nodepath = self.plot_surface_lines(lons,lats, **args)
        return nodepath

    def add_coastline(self, res = '50m', **args):
        np_list = []
        if res == '50m':
            sf = shapefile.Reader('./resources/ne_50m_coastline.shp')
        elif res == '110m':
            sf = shapefile.Reader('./resources/ne_110m_coastline.shp')
        elif res == '10m':
            sf = shapefile.Reader('./resources/ne_10m_coastline.shp')
        else:
            raise ValueError('res must be either 50m or 110m.')
        shapes = sf.shapes()
        for s in shapes:
            np_list.append(self.plot_shape(s, **args))
        return np_list


if __name__ == '__main__':
    # plot = EarthPlot(d_light_strength = (.3,.3,.3,1))
    # datasize = 10000
    # lons2 = np.linspace(0,40,50)
    # lats2 = lons2
    # lons = np.concatenate(((np.random.randn(datasize)-.5)*10 , (np.random.randn(datasize)-.5)*10 + np.random.rand(1)*80))
    # lats = np.concatenate(((np.random.randn(datasize)-.5)*10 , (np.random.randn(datasize)-.5)*10 + np.random.rand(1)*80))
    # plot.plot_heatmap(lons,lats, n_bins = 200, texture_filter = 'linear', alpha = .7, mpl_cm = 'jet')
    # plot.add_coastline(res = '50m', color =[0,0,0], linewidth = 2, shading = 'None')
    # plot.plot_geodetic(-170,0, 170,0, color = [1,0,0])
    # plot.show()
    plot = EarthPlot()
    datasize = 10000
    lons2 = np.linspace(0,40,50)
    lats2 = lons2
    lons = np.concatenate(((np.random.randn(datasize)-.5)*10 , (np.random.randn(datasize)-.5)*10 + np.random.rand(1)*80))
    lats = np.concatenate(((np.random.randn(datasize)-.5)*10 , (np.random.randn(datasize)-.5)*10 + np.random.rand(1)*80))
    plot.plot_heatmap(lons,lats, n_bins = 200, texture_filter = 'linear', alpha = .7, mpl_cm = 'jet')
    plot.add_coastline(res = '50m', color =[0,0,0], linewidth = 2, shading = 'None')
    plot.plot_geodetic(-170,0, 170,0, color = [1,0,0])
    app = QApplication(sys.argv)
    appw = QMainWindow()
    appw.setGeometry(50, 50, 800, 600)

    pandaWidget = QPanda3DWidget_(plot)

    appw.setCentralWidget(pandaWidget)
    appw.show()

    sys.exit(app.exec_())
