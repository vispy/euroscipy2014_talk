import itertools
import os
import hashlib

import numpy as np
import vispy
from vispy import app
from vispy import scene
from vispy import gloo

from demovisuals import Video, Rain, Boids, Atom, RealtimeSignals, Raycasting

import imageio

# Shorthands
from vispy.scene.visuals import Text
from vispy.scene.visuals import Image


LOAD_IMAGES_FROM_CACHE = True
THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def load_image(url):
    """ To read images/diagrams from GDrive and cahche them.
    """
    if not '/' in url:
        url = os.path.join(THIS_DIR, 'images', url)
    if url.startswith('http'):
        fname = hashlib.md5(url.encode('utf-8')).hexdigest() + '.png'
    else:
        fname = url.split('/')[-1]
    filename = os.path.join(THIS_DIR, 'images', fname)
    if filename == url:
        return imageio.imread(filename)
    elif not LOAD_IMAGES_FROM_CACHE:
        data = imageio.imread(url, os.path.splitext(fname)[1])
        imageio.imsave(filename, data)
        return data
    elif os.path.isfile(filename):
        return imageio.imread(filename)
    else:
        return np.zeros((10, 10), np.uint8)


class Presentation(scene.SceneCanvas):
    """ Presentation canvas.
    """
    
    def __init__(self, title='A presentation', **kwargs):
        self._current_slide = 0
        self._slides = []
        scene.SceneCanvas.__init__(self, title=title, bgcolor='white', **kwargs)
        self.size = 800, 600
        refresh = lambda x: self.update()
        self._timer = app.Timer(1/30, connect=refresh, start=True)
        
        # Create toplevel viewbox that we can use to see whole presentation
        self._top_vb = scene.widgets.ViewBox(parent=self.scene, border_color=None)
        #self._top_vb.camera = scene.cameras.PerspectiveCamera('perspective')
        #self._top_vb.camera.fov = 80
        self._top_vb.camera = scene.cameras.TurntableCamera()
        self._top_vb.camera.up = 'y'
        self._top_vb.camera.center = -2200, +100, -4000
        self._top_vb.camera.width = 8000
        self._top_vb.camera.azimuth = -30
        self._top_vb.camera.elevation = 30
        self._top_vb.camera._update_camera_pos()  # todo: camera must do this
        self._top_vb.size = self.size
        
        self.show_text = True
    
    def on_initialize(self, event):
        gloo.set_state('translucent',
        #blend=True,
                       #blend_func=('src_alpha', 'one_minus_src_alpha'),
                       depth_test=False)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *self.size)
        for slide in self._slides:
            expand = self.size[0] / self.size[1]
            slide.camera.rect = -0.5*(expand-1), 0, expand, 1
            slide.size = self.size
    
    def add_slide(self, title=None, cls=None):
        # todo: make _children a list or ordered set
        # Create super viewbox
        super_vb = scene.widgets.ViewBox(parent= self._top_vb.scene, border_color='b')
        super_vb.transform = scene.transforms.AffineTransform()
        nx = len(self._slides) % 8
        ny = len(self._slides) // 8
        for i in range(nx):
            super_vb.transform.rotate(-10, (0, 1, 0))
            super_vb.transform.translate((1000, 0))
        super_vb.transform.translate((0, -1000*ny))
        super_vb.size = self.size
        super_vb.camera = scene.cameras.BaseCamera()  # Null camera
        
        # Create slide itself
        cls = cls or Slide
        slide = cls(parent=super_vb.scene, border_color=None)
        slide.camera.interactive = False
        slide.camera.invert_y = False
        expand = self.size[0] / self.size[1]
        slide.camera.rect = 0, 0, 1, 1  # overwritten on resize
        self._slides.append(slide)
        slide.size = self.size
        slide.clip_method = 'viewport'
        slide.events.add(key_press=None)
        slide.scene._systems['draw'] = DrawingSystem()
        if title:
            slide.set_title(title)
        # Select this slide now?
        if len(self._slides) == 1:
            self.scene = self._slides[-1].superviewbox.scene
        return slide
    
    def on_key_press(self, event):
        if event.key == vispy.keys.HOME:
            self.scene = self._top_vb.parent
        elif event.key in (vispy.keys.LEFT, vispy.keys.RIGHT, vispy.keys.END, vispy.keys.SPACE):
            for child in self._slides[self._current_slide].scene.children:
                if hasattr(child, 'on_leave_slide'): child.on_leave_slide()
            M = {vispy.keys.LEFT:-1, vispy.keys.RIGHT:1, vispy.keys.SPACE:1}
            self._current_slide += M.get(event.key, 0)
            self._current_slide = max(min(self._current_slide, len(self._slides)-1), 0)
            self.scene = self._slides[self._current_slide].superviewbox.scene
            for child in self._slides[self._current_slide].scene.children:
                if hasattr(child, 'on_enter_slide'): child.on_enter_slide()
        elif event.text.lower() == 'f':
            self.fullscreen = not self.fullscreen
            self.on_resize(None)
        else:
            self._slides[self._current_slide].events.key_press(event)
        self.show_text = self.scene is not self._top_vb.parent


class Slide(scene.widgets.ViewBox):
    """ Representation of a slide in the presentation in the form of a viewbox.
    """
    
    @property
    def superviewbox(self):
        return self.parent.parent
    
    def set_title(self, text):
        text = Text(text, parent=self.scene, pos=(0.5, 0.1), 
                    font_size=26, bold=True, color='#002233', 
                    font_manager=font_manager)
        return text
    
    def add_text(self, text, pos, size=20, anchor_x='left', **kwargs):
        text = Text(text, parent=self.scene, pos=pos, font_size=size, 
                    anchor_x=anchor_x, anchor_y='center',
                    font_manager=font_manager, **kwargs)
        return text
    
    def add_image(self, url, pos, size, **kwargs):
        # size is for width
        data = load_image(url)
        im = Image(data)
        im.add_parent(self.scene)
        im.transform = scene.transforms.STTransform()
        im.transform.scale = size / data.shape[1], size / data.shape[1]
        im.transform.translate = pos
        im.interpolation = gloo.gl.GL_LINEAR
        # todo: make Image Visual that we can position/scale
        return im


class DrawingSystem(scene.systems.DrawingSystem):
    def _process_entity(self, event, entity, force_recurse=False):
        if (not pres.show_text) and isinstance(entity, scene.visuals.Text):
            return
        else:
            scene.systems.DrawingSystem._process_entity(self, event, entity, force_recurse)


# For more efficient fonts
font_manager = scene.visuals.text.text.FontManager()

pres = Presentation(show=True)

## Introduction


slide = pres.add_slide()
# slide.add_image('vispylogo.png', (0.42, 0.05), 0.16)
slide.add_text("Introducing Vispy's high level modules:", (0.5, 0.5), 24, 'center', color='k', bold=True)
slide.add_text("easy yet powerful visualization for everyone", (0.5, 0.6), 24, 'center', color='k', bold=True)
slide.add_text("EuroScipy - August 30, 2014", (0.5, 0.8), 20, 'center', color='b', bold=True)
t1 = slide.add_text("Almar Klein", (0.5, 0.9), 20, 'center', color='k')

atom = Atom(parent=slide.scene)
atom._timer.start()  # manual start

slide = pres.add_slide("Purpose")
t2 = slide.add_text("  • Data is growing (e.g. 3D data or big data sets)", (0, 0.3), )
t2 = slide.add_text("  • Don't just look, explore!", (0, 0.35), )
slide.add_image('Astronomy_Amateur_3_V2.jpg', (0, 0.5), 0.4)
slide.add_image('Astronaut_moon_rock.jpg', (0.6, 0.5), 0.4)

slide = pres.add_slide("Premise")
slide.add_text("How to achieve speed", (0, 0.25), bold=True )
slide.add_text("  • Use power of GPU", (0, 0.3), )
slide.add_text("  • Custom shaders for quality", (0, 0.35), )
slide.add_image('gpu.jpg', (0.4, 0.6), 0.4)


slide = pres.add_slide("Goals")
slide.add_text("  • Easy to install (Pure Python)", (0, 0.3), )
slide.add_text("  • Easy to use, yet flexible", (0, 0.35), )
slide.add_text("  • Interactive & fast, yet high quality", (0, 0.40), )
slide.add_text("  • Support for embedding in apps", (0, 0.45), )


slide = pres.add_slide("About Vispy")

slide.add_text("Team: ", (0, 0.25), bold=True)
slide.add_text("  • Luke Campagnola (pyqtgraph)", (0, 0.3), )
slide.add_text("  • Almar Klein (visvis)", (0, 0.35), )
slide.add_text("  • Eric Larson", (0, 0.4), )
slide.add_text("  • Cyrille Rossant (Galry)", (0, 0.45), )
slide.add_text("  • Nicolas Rougier (Glumpy)", (0, 0.5), )
slide.add_text("  • Several contributors ", (0, 0.55), )

slide.add_text("Facts: ", (0, 0.65), bold=True )
slide.add_text("  • http://vispy.org", (0, 0.70), )
slide.add_text("  • http://github.com/vispy/vispy", (0, 0.75), )
slide.add_text("  • CI: flake8, tests", (0, 0.8), )


## Inside vispy

slide = pres.add_slide("Vispy structure")
slide.add_image('https://docs.google.com/drawings/d/1Is9V51XtU7eRvru1X8jceLT2PoRsiSmCs6Jcaa9ojMQ/pub?w=960', (0.1, 0.15), 0.8)



## vispy.app

# slide = pres.add_slide()
# slide.add_text("App", (0.5, 0.5), 32, 'center', color='blue')
# 
# slide = pres.add_slide("vispy.app")
# slide.add_text("  • Provides Canvas, Timer, mainloop", (0, 0.2), )
# slide.add_text("  • Backends: PyQt4, Pyside, WX, Pyglet, Glfw, Sdl2, Glut", (0, 0.25), )
# slide.add_text("  ... and IPython notebook!", (0, 0.3), )
# slide.add_image('code_app.png', (0.1, 0.4), 0.8)


## vispy.gloo


slide = pres.add_slide()
slide.add_text("Gloo", (0.5, 0.5), 32, 'center', color='blue')

slide = pres.add_slide("vispy.gloo")

slide.add_text("  • Object oriented API for OpenGL", (0, 0.2), )
slide.add_text("  • Classes: Program, VertexShader, FragmentShader, ", (0, 0.3), )
slide.add_text("    VertexBuffer, IndexBuffer, Texture2D, ", (0, 0.35), )
slide.add_text("  • Functions: clear(), set_viewport(), etc. ", (0, 0.45), )

slide = pres.add_slide("Gloo demo - rain")
rain = Rain(parent=slide.scene)

slide = pres.add_slide("Gloo demo - raycasting")
raycasting = Raycasting(parent=slide.scene)


## vispy.visuals

slide = pres.add_slide()
slide.add_text("Visuals", (0.5, 0.5), 32, 'center', color='blue')

slide = pres.add_slide("Visuals")

slide.add_text("  • Pythonic (high-level) objects", (0, 0.2), )
slide.add_text("  • Large collection in vispy", (0, 0.25), )
slide.add_text("  • Extensible: create your own visual", (0, 0.3), )
slide.add_image('https://docs.google.com/drawings/d/1OIkoeI7NrOdZx5MI5IZB4zMXbOs0Cy9RE97mLyLxMII/pub?w=840', (0.1, 0.4), 0.8)

slide = pres.add_slide("Visual demo - video")
vid = Video((0.0, 0.4), 1.0, parent=slide.scene)
slide.events.key_press.connect(vid.swap_channel)


## vispy.scene

slide = pres.add_slide()
slide.add_text("Scene", (0.5, 0.5), 32, 'center', color='blue')

slide = pres.add_slide("vispy.scene (1)")

slide.add_text("Base object: Entity", (0, 0.2), bold=True)
slide.add_text("  • has transform", (0, 0.25), )
slide.add_text("  • can have 0 or more children", (0, 0.3), )
slide.add_text("  • can have 0 or more parents", (0, 0.35), )

slide.add_text("Important subclasses:", (0, 0.5), bold=True)
slide.add_text("  • Visual", (0, 0.55), )
slide.add_text("  • Widget", (0, 0.6), )
slide.add_text("  • ViewBox", (0, 0.65), )
slide.add_text("  • Camera", (0, 0.7), )

slide = pres.add_slide("vispy.scene (2)")
slide.add_image('https://docs.google.com/drawings/d/1U4Ym8VhIi4CVftCf5wl1POzOryoP4kD9nhNIH7HPlD4/pub?w=1000', (-0.1, 0.2), 1.2)


slide = pres.add_slide("vispy.scene (3)")
slide.add_image('code_scene.png', (0.0, 0.2), 1.0)


class SceneSlide(Slide):
    def __init__(self, *args, **kwargs):
        Slide.__init__(self, *args, **kwargs)
        self._vb = vb = scene.widgets.ViewBox(parent=self.scene, size=(0.5, 0.3), border_color='blue')
        vb.pos = 0.2, 0.25
        vb.camera.rect = 0, 0, 1, 1
        #
        line_data = np.random.normal(0.5, 0.2, size=(100, 2)).astype('float32')
        line_data[:,0] = np.linspace(0, 1, 100)
        line = scene.visuals.Line(pos=line_data, color=(1,0,0,1))
        line.parent = vb.scene
        
    def on_key_press(self, event):
        if event.key == vispy.keys.DOWN:
            self.camera.rect = self._vb.pos, self._vb.size
            #self.superviewbox.parent.parent = self._vb.scene
        elif event.key == vispy.keys.UP:
            pres.on_resize(None)
            #self.superviewbox.parent.parent = self.superviewbox.scene

slide = pres.add_slide("vispy.scene (4)", SceneSlide)
slide.add_text("Here is a viewbox ... ", (0.1, 0.2), bold=True)
slide.add_text("And this slide is a ViewBox too ... ", (0.1, 0.7), bold=True)
slide.camera.interactive = True

slide = pres.add_slide("Scene demo - realtime signals ")
realtimesignals = RealtimeSignals(parent=slide.scene)
realtimesignals.set_transform('st', translate=(0, 0.2))

slide = pres.add_slide("A use case of scenegraph: VR/AR")
slide.add_text("  • Stereo images (same scene in two ViewBoxes)", (0, 0.2), )
slide.add_text("  • Viewpoint (just move camera object)", (0, 0.25), )
slide.add_image('hmd_open.png', (0.1, 0.4), 0.8)


## mpl_plot and plot

slide = pres.add_slide()
slide.add_text("Functional interfaces", (0.5, 0.5), 32, 'center', color='blue')

slide = pres.add_slide("Functional plotting interfaces")
slide.add_text("vispy.mpl_plot", (0, 0.3), bold=True)
slide.add_text("  • Fully compatible with mpl.pyplot", (0, 0.35), )
slide.add_text("  • Requires MPL", (0, 0.4), )

slide.add_text("vispy.plot:", (0, 0.5), bold=True)
slide.add_text("  • ~ compatible with mpl.pyplot", (0, 0.55), )
slide.add_text("  • No dependencies", (0, 0.6), )
slide.add_text("  • More flexibility", (0, 0.65), )


slide.add_text("Other interfaces?", (0, 0.75), bold=True)
slide.add_text("  • Grammar of Graphics / ggplot", (0, 0.8), )

slide = pres.add_slide("Functional plotting interfaces")
slide.add_image('code_mpl_plot.png', (0.1, 0.2), 0.8)


## IPython

slide = pres.add_slide()
slide.add_text("IPython notebook", (0.5, 0.5), 32, 'center', color='blue')

slide = pres.add_slide("Rendering in the browser")
slide.add_image('https://docs.google.com/drawings/d/1MbX1MN-0KGoFuq-eQTByhtIxpqvHkMIWJNWERUaFwfo/pub?w=900', (0.0, 0.2), 1.0)


## Almost done
 
slide = pres.add_slide()
slide.add_text("Wrapping up", (0.5, 0.5), 32, 'center', color='blue')

slide = pres.add_slide("Current status")
 
slide.add_text("  • vispy.app and vispy.gloo pretty stable API", (0, 0.2), )
slide.add_text("  • vispy.scene ~ alpha status", (0, 0.25), )
slide.add_text("  • also need picking, transparancy, collections, ...", (0, 0.3), )
slide.add_text("  • vispy.mpl_plot experimental, but API stable", (0, 0.35), )

slide.add_text("Sprint tomorrow!", (0, 0.45), )

slide = pres.add_slide("Thank you!")
slide.add_text("http://vispy.org", (0.5, 0.45), 16, 'center', color='b', bold=True)

boids = Boids(parent=slide.scene)


if __name__ == '__main__':
    pres.on_resize(None)
    pres.show()
    pres.app.run()
