import os
import time
import math

import numpy as np
from scipy.spatial import cKDTree
import imageio

import vispy
from vispy import app, scene, gloo
from vispy.scene.shaders import ModularProgram


class SlideVisualMixin:
    
    def on_enter_slide(self, event=None):  # called by pres
        self._timer.start()
    
    def on_leave_slide(self, event=None):  # called by pres
        self._timer.stop()
    
class Video(scene.visuals.Image, SlideVisualMixin):
    
    def __init__(self, pos, size, **kwargs):
        
        reader1 = imageio.read('<video0>', 'ffmpeg')
        reader2 = imageio.read(os.path.join(os.path.expanduser('~'), 'Videos', 'ice age 4 trailer.mp4'), 'ffmpeg', loop=True)
        reader2.get_data(5*30)
        self._timer = app.Timer(1.0/30, connect=self.next_image)
        
        self._readers = reader1, reader2
        self._reader = reader2
        
        im = self._reader.get_next_data()
        scene.visuals.Image.__init__(self, im, **kwargs)
        self.transform = scene.transforms.STTransform()
        self.transform.scale = size / im.shape[1], size / im.shape[1]
        self.transform.translate = pos
    
    def next_image(self, event):
        im = self._reader.get_next_data()
        self.set_data(im)
    
    def swap_channel(self, event=None):
        if event is None or event.key in [vispy.keys.UP, vispy.keys.DOWN]:
            if self._reader == self._readers[0]:
                self._reader = self._readers[1]
            else:
                self._reader = self._readers[0]


class Rain(scene.visuals.Visual, SlideVisualMixin):
    vertex = """
        #version 120
        
        uniform float u_linewidth;
        uniform float u_antialias;
        
        attribute vec2  a_position;
        attribute vec4  a_fg_color;
        attribute float a_size;
        
        varying vec4  v_fg_color;
        varying float v_size;
        
        void main (void)
        {
            v_size = a_size;
            v_fg_color = a_fg_color;
            if( a_fg_color.a > 0.0)
            {
                gl_Position = $transform(vec4(a_position, 0.0, 1.0));
                gl_PointSize = v_size + u_linewidth + 2*1.5*u_antialias;
            }
            else
            {
                gl_Position = $transform(vec4(-1.0, -1.0, 0.0, 1.0));
                gl_PointSize = 0.0;
            }
        }
        """
        
    fragment = """
        #version 120
        
        uniform float u_linewidth;
        uniform float u_antialias;
        varying vec4  v_fg_color;
        varying vec4  v_bg_color;
        varying float v_size;
        float disc(vec2 P, float size)
        {
            return length((P.xy - vec2(0.5,0.5))*size);
        }
        void main()
        {
            if( v_fg_color.a <= 0.0)
                discard;
            float actual_size = v_size + u_linewidth + 2*1.5*u_antialias;
            float t = u_linewidth/2.0 - u_antialias;
            float r = disc(gl_PointCoord, actual_size);
            float d = abs(r - v_size/2.0) - t;
            if( d < 0.0 )
            {
                gl_FragColor = v_fg_color;
            }
            else if( abs(d) > 2.5*u_antialias )
            {
                discard;
            }
            else
            {
                d /= u_antialias;
                gl_FragColor = vec4(v_fg_color.rgb, exp(-d*d)*v_fg_color.a);
            }
        }
        """
    
    def __init__(self, **kwargs):
        scene.visuals.Visual.__init__(self, **kwargs)
        
        self._n = 250
        self.data = np.zeros(self._n, [('a_position', np.float32, 2),
                                 ('a_fg_color', np.float32, 4),
                                 ('a_size',     np.float32, 1)])
        self.index = 0
        self.program = ModularProgram(self.vertex, self.fragment)
        self.vdata = gloo.VertexBuffer(self.data)
        self._timer = app.Timer(1. / 60., self.on_timer)
    
    def draw(self, event):
        xform = event.render_transform.shader_map()
        self.program.vert['transform'] = xform
        
        self.program.prepare()  
        self.program.bind(self.vdata)
        self.program['u_antialias'] = 1.00
        self.program['u_linewidth'] = 2.00
        
        self.program.draw('points')
    
    def on_timer(self, event):
        self.data['a_fg_color'][..., 3] -= 0.01
        self.data['a_size'] += 1.0
        self.vdata.set_data(self.data)

    def on_mouse_move(self, event):
        x, y = event.pos[:2]
        #h = gloo.get_parameter('viewport')[3]
        self.data['a_position'][self.index] = x, y
        self.data['a_size'][self.index] = 5
        self.data['a_fg_color'][self.index] = 0, 0, 0, 1
        self.index = (self.index + 1) % self._n


class Boids(scene.visuals.Visual, SlideVisualMixin):
    
    VERT_SHADER = """
        #version 120
        attribute vec3 position;
        attribute vec4 color;
        attribute float size;
        
        varying vec4 v_color;
        void main (void) {
            gl_Position = $transform(vec4(position, 1.0));
            v_color = color;
            gl_PointSize = size;
        }
        """
    
    FRAG_SHADER = """
        #version 120
        varying vec4 v_color;
        void main()
        {
            float x = 2.0*gl_PointCoord.x - 1.0;
            float y = 2.0*gl_PointCoord.y - 1.0;
            float a = 1.0 - (x*x + y*y);
            gl_FragColor = vec4(v_color.rgb, a*v_color.a);
        }
        """

    def __init__(self, **kwargs):
        scene.visuals.Visual.__init__(self, **kwargs)
        
        # Create boids
        n = 1000
        particles = np.zeros(2 + n, [('position', 'f4', 3),
                                    ('position_1', 'f4', 3),
                                    ('position_2', 'f4', 3),
                                    ('velocity', 'f4', 3),
                                    ('color', 'f4', 4),
                                    ('size', 'f4', 1)])
        boids = particles[2:]
        target = particles[0]
        predator = particles[1]
        
        boids['position'] = np.random.uniform(0, 1, (n, 3))
        boids['velocity'] = np.random.uniform(-0.00, +0.00, (n, 3))
        boids['size'] = 4
        boids['color'] = 0.4, 0.4, 0.8, 1
        
        target['size'] = 16
        target['color'][:] = 0, 0, 1, 1
        predator['size'] = 16
        predator['color'][:] = 1, 0, 0, 1
        target['position'][:] = 0.25, 0.5, 0
        predator['position'][:] = 0.7, 0.4, 0
        
        self._boids, self._target, self._predator = boids, target, predator
        self._particles = particles
        
        # Time
        self._t = time.time()
        self._pos = 0.0, 0.0
        self._button = None

        # Create program
        self.program = ModularProgram(self.VERT_SHADER, self.FRAG_SHADER)

        # Create vertex buffers
        self.vbo_position = gloo.VertexBuffer(particles['position'].copy())
        self.vbo_color = gloo.VertexBuffer(particles['color'].copy())
        self.vbo_size = gloo.VertexBuffer(particles['size'].copy())
        
        self._timer = app.Timer(1.0/30, self.iteration)
    
    def on_mouse_press(self, event):
        self._button = event.button
        self.on_mouse_move(event)

    def on_mouse_release(self, event):
        self._button = None
        self.on_mouse_move(event)

    def on_mouse_move(self, event):
        if not self._button:
            return
        #w, h = self.size
        x, y = event.pos[:2]
        sx, sy = x, y
        #sx = 2 * x / float(w) - 1.0
        #sy = - (2 * y / float(h) - 1.0)

        if self._button == 1:
            self._target['position'][:] = sx, sy, 0
        elif self._button == 2:
            self._predator['position'][:] = sx, sy, 0

    def draw(self, event):
        
        # Set transform
        xform = event.render_transform.shader_map()
        self.program.vert['transform'] = xform
        self.program.prepare()  
        
        # Bind vertex buffers
        self.program['color'] = self.vbo_color
        self.program['size'] = self.vbo_size
        self.program['position'] = self.vbo_position
        
        # Draw
        self.program.draw('points')
    

    def iteration(self, event=None):
        
        boids, target, predator = self._boids, self._target, self._predator

        t = time.time()
        target['position'][:] = np.sin(t*0.3)*0.4+0.5, 0.5, 0
        
        #t += 0.5 * dt
        #predator[...] = np.array([np.sin(t),np.sin(2*t),np.cos(3*t)])*.2

        boids['position_2'] = boids['position_1']
        boids['position_1'] = boids['position']
        n = len(boids)
        P = boids['position']
        V = boids['velocity']

        # Cohesion: steer to move toward the average position of local
        # flockmates
        C = -(P - P.sum(axis=0) / n)

        # Alignment: steer towards the average heading of local flockmates
        A = -(V - V.sum(axis=0) / n)

        # Repulsion: steer to avoid crowding local flockmates
        D, I = cKDTree(P).query(P, 5)
        M = np.repeat(D < 0.05, 3, axis=1).reshape(n, 5, 3)
        Z = np.repeat(P, 5, axis=0).reshape(n, 5, 3)
        R = -((P[I] - Z) * M).sum(axis=1)

        # Target : Follow target
        T = target['position'] - P

        # Predator : Move away from predator
        dP = P - predator['position']
        D = np.maximum(0, 0.3 -
                       np.sqrt(dP[:, 0] ** 2 +
                               dP[:, 1] ** 2 +
                               dP[:, 2] ** 2))
        D = np.repeat(D, 3, axis=0).reshape(n, 3)
        dP *= D

        #boids['velocity'] += 0.0005*C + 0.01*A + 0.01*R + 0.0005*T + 0.0025*dP
        boids['velocity'] += 0.0005 * C + 0.01 * \
            A + 0.01 * R + 0.0005 * T + 0.025 * dP
        boids['position'] += boids['velocity']

        self.vbo_position.set_data(self._particles['position'])

        return t


class Atom(scene.visuals.Visual, SlideVisualMixin):
        
    vert = """
        #version 120
        uniform float u_size;
        uniform float u_clock;
        
        attribute vec2 a_position;
        attribute vec4 a_color;
        attribute vec4 a_rotation;
        varying vec4 v_color;
        
        mat4 build_rotation(vec3 axis, float angle)
        {
            axis = normalize(axis);
            float s = sin(angle);
            float c = cos(angle);
            float oc = 1.0 - c;
            return mat4(oc * axis.x * axis.x + c,
                        oc * axis.x * axis.y - axis.z * s,
                        oc * axis.z * axis.x + axis.y * s,
                        0.0,
                        oc * axis.x * axis.y + axis.z * s,
                        oc * axis.y * axis.y + c,
                        oc * axis.y * axis.z - axis.x * s,
                        0.0,
                        oc * axis.z * axis.x - axis.y * s,
                        oc * axis.y * axis.z + axis.x * s,
                        oc * axis.z * axis.z + c,
                        0.0,
                        0.0, 0.0, 0.0, 1.0);
        }
        
        
        void main (void) {
            v_color = a_color;
        
            float x0 = 1.5;
            float z0 = 0.0;
        
            float theta = a_position.x + u_clock;
            float x1 = x0*cos(theta) + z0*sin(theta);
            float y1 = 0.0;
            float z1 = (z0*cos(theta) - x0*sin(theta))/2.0;
            
        
            mat4 R = build_rotation(a_rotation.xyz, a_rotation.w);
            vec4 pos = R * vec4(x1,y1,z1,1);
            pos.x = pos.x * 0.13 + 0.5;
            pos.y = pos.y * 0.13 + 0.22;
            gl_Position = $transform(pos);
            gl_PointSize = 12.0 * u_size * sqrt(v_color.a);
        }
        """
        
    frag = """
        #version 120
        varying vec4 v_color;
        varying float v_size;
        void main()
        {
            float d = 2*(length(gl_PointCoord.xy - vec2(0.5,0.5)));
            gl_FragColor = vec4(v_color.rgb, v_color.a*(1-d));
        }
        """

    def __init__(self, **kwargs):
        scene.visuals.Visual.__init__(self, **kwargs)
        
        # Create vertices
        n, p = 150, 32
        data = np.zeros(p * n, [('a_position', np.float32, 2),
                                ('a_color',    np.float32, 4),
                                ('a_rotation', np.float32, 4)])
        trail = .5 * np.pi
        data['a_position'][:, 0] = np.resize(np.linspace(0, trail, n), p * n)
        data['a_position'][:, 0] += np.repeat(np.random.uniform(0, 2 * np.pi, p), n)
        data['a_position'][:, 1] = np.repeat(np.linspace(0, 2 * np.pi, p), n)
        
        data['a_color'] = 1, 1, 1, 1
        data['a_color'] = np.repeat(
            np.random.uniform(0.5, 1.00, (p, 4)).astype(np.float32), n, axis=0)
        data['a_color'][:, 3] = np.resize(np.linspace(0, 1, n), p * n)
        
        data['a_rotation'] = np.repeat(
            np.random.uniform(0, 2 * np.pi, (p, 4)).astype(np.float32), n, axis=0)
            
       
        self.program = ModularProgram(self.vert, self.frag)
        self._vbo = gloo.VertexBuffer(data)
        
        self.theta = 0
        self.phi = 0
        self.clock = 0
        self.stop_rotation = False
        
        self.transform = vispy.scene.transforms.AffineTransform()

        self._timer = app.Timer(1.0 / 30, self.on_timer)

    def on_timer(self, event):
        self.clock += np.pi / 100

    def draw(self, event):
        # Set transform
        xform = event.render_transform.shader_map()
        self.program.vert['transform'] = xform
        self.program.prepare()  
        
        # Bind variables
        self.program.bind(self._vbo)
        self.program['u_size'] = 5 / 6
        self.program['u_clock'] = self.clock

        self.program.draw('points')



class RealtimeSignals(scene.visuals.Visual, SlideVisualMixin):
    
    VERT_SHADER = """
        #version 120
        
        // y coordinate of the position.
        attribute float a_position;
        
        // row, col, and time index.
        attribute vec3 a_index;
        varying vec3 v_index;
        
        // 2D scaling factor (zooming).
        uniform vec2 u_scale;
        
        // Size of the table.
        uniform vec2 u_size;
        
        // Number of samples per signal.
        uniform float u_n;
        
        // Color.
        attribute vec3 a_color;
        varying vec4 v_color;
        
        // Varying variables used for clipping in the fragment shader.
        varying vec2 v_position;
        varying vec4 v_ab;
        
        void main() {
            float nrows = u_size.x;
            float ncols = u_size.y;
        
            // Compute the x coordinate from the time index.
            //float x = -1 + 2*a_index.z / (u_n-1);
            float x = a_index.z / (u_n-1);
            vec2 position = vec2(x, a_position);
            
            // Find the affine transformation for the subplots.
            vec2 a = vec2(1./ncols, 1./nrows)*.9;
            vec2 b = vec2((a_index.x+.5) / ncols, 
                          (a_index.y+.5) / nrows);
            // Apply the static subplot transformation + scaling.
            vec4 abs_position = vec4(a*u_scale*position+b, 0.0, 1.0);
            gl_Position = $transform(abs_position);
            
            v_color = vec4(a_color, 1.);
            v_index = a_index;
            
            // For clipping test in the fragment shader.
            v_position = abs_position.xy;
            v_ab = vec4(a, b);
        }
        """
        
    FRAG_SHADER = """
        #version 120
        
        varying vec4 v_color;
        varying vec3 v_index;
        
        varying vec2 v_position;
        varying vec4 v_ab;
        
        void main() {
            gl_FragColor = v_color;
            
            // Discard the fragments between the signals (emulate glMultiDrawArrays).
            //if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
            //    discard;
            
            // Clipping test.
            //vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
            //if ((test.x > 1) || (test.y > 1))
             //   discard;
        }
        """

    def __init__(self, **kwargs):
        scene.visuals.Visual.__init__(self, **kwargs)
        
        # Number of cols and rows in the table.
        self._nrows = nrows = 16
        self._ncols = ncols = 20
        # Number of signals.
        self._m = m = nrows*ncols
        # Number of samples per signal.
        self._n = n = 1000
        # Various signal amplitudes.
        amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)
        self._amplitudes = amplitudes
        
        # Generate the signals as a (m, n) array.
        self._y = y = amplitudes * np.random.randn(m, n).astype(np.float32)
        
        # Color of each vertex (TODO: make it more efficient by using a GLSL-based 
        # color map and the index).
        color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                        n, axis=0).astype(np.float32)
        
        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
                    np.repeat(np.tile(np.arange(nrows), ncols), n),
                    np.tile(np.arange(n), m)].astype(np.float32)

        self.program = ModularProgram(self.VERT_SHADER, self.FRAG_SHADER)
        self._pos_vbo = gloo.VertexBuffer(y.ravel())
        self._color_vbo = gloo.VertexBuffer(color)
        self._index_vbo = gloo.VertexBuffer(index)
        
        self._timer = app.Timer('auto', connect=self.on_timer)

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        k = 10
        self._y[:, :-k] = self._y[:, k:]
        self._y[:, -k:] = self._amplitudes * np.random.randn(self._m, k)
        
        self._pos_vbo.set_data(self._y.ravel().astype(np.float32))
    
    def draw(self, event):
        # Set transform
        xform = event.render_transform.shader_map()
        self.program.vert['transform'] = xform
        self.program.prepare()  
        
        # Bind variables
        self.program['a_position'] = self._pos_vbo
        self.program['a_color'] = self._color_vbo
        self.program['a_index'] = self._index_vbo
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (self._nrows, self._ncols)
        self.program['u_n'] = self._n
        
        self.program.draw('line_strip')


class Raycasting(scene.Visual, SlideVisualMixin):
    
    vertex = """
        #version 120
        
        attribute vec2 a_position;
        varying vec2 v_position;
        void main()
        {
            gl_Position = $transform(vec4(a_position, 0.0, 1.0));
            v_position = gl_Position.xy;
        }
        """
        
    fragment = """
        #version 120
        
        const float M_PI = 3.14159265358979323846;
        const float INFINITY = 1000000000.;
        const int PLANE = 1;
        const int SPHERE_0 = 2;
        const int SPHERE_1 = 3;
        
        uniform float u_time;
        uniform float u_aspect_ratio;
        varying vec2 v_position;
        
        uniform vec3 sphere_position_0;
        uniform float sphere_radius_0;
        uniform vec3 sphere_color_0;
        
        uniform vec3 sphere_position_1;
        uniform float sphere_radius_1;
        uniform vec3 sphere_color_1;
        
        uniform vec3 plane_position;
        uniform vec3 plane_normal;
        
        uniform float light_intensity;
        uniform vec2 light_specular;
        uniform vec3 light_position;
        uniform vec3 light_color;
        
        uniform float ambient;
        uniform vec3 O;
        
        float intersect_sphere(vec3 O, vec3 D, vec3 S, float R) {
            float a = dot(D, D);
            vec3 OS = O - S;
            float b = 2. * dot(D, OS);
            float c = dot(OS, OS) - R * R;
            float disc = b * b - 4. * a * c;
            if (disc > 0.) {
                float distSqrt = sqrt(disc);
                float q = (-b - distSqrt) / 2.0;
                if (b >= 0.) {
                    q = (-b + distSqrt) / 2.0;
                }
                float t0 = q / a;
                float t1 = c / q;
                t0 = min(t0, t1);
                t1 = max(t0, t1);
                if (t1 >= 0.) {
                    if (t0 < 0.) {
                        return t1;
                    }
                    else {
                        return t0;
                    }
                }
            }
            return INFINITY;
        }
        
        float intersect_plane(vec3 O, vec3 D, vec3 P, vec3 N) {
            float denom = dot(D, N);
            if (abs(denom) < 1e-6) {
                return INFINITY;
            }
            float d = dot(P - O, N) / denom;
            if (d < 0.) {
                return INFINITY;
            }
            return d;
        }
        
        vec3 run(float x, float y, float t) {
            vec3 Q = vec3(x, y, 0.);
            vec3 D = normalize(Q - O);
            int depth = 0;
            float t_plane, t0, t1;
            vec3 rayO = O;
            vec3 rayD = D;
            vec3 col = vec3(0.0, 0.0, 0.0);
            vec3 col_ray;
            float reflection = 1.;
            
            int object_index;
            vec3 object_color;
            vec3 object_normal;
            float object_reflection;
            vec3 M;
            vec3 N, toL, toO;
            
            while (depth < 5) {
                
                /* start trace_ray */
                
                t_plane = intersect_plane(rayO, rayD, plane_position, plane_normal);
                t0 = intersect_sphere(rayO, rayD, sphere_position_0, sphere_radius_0);
                t1 = intersect_sphere(rayO, rayD, sphere_position_1, sphere_radius_1);
                
                if (t_plane < min(t0, t1)) {
                    // Plane.
                    M = rayO + rayD * t_plane;
                    object_normal = plane_normal;
                    // Plane texture.
                    if (mod(int(2*M.x), 2) == mod(int(2*M.z), 2)) {
                        object_color = vec3(1., 1., 1.);
                    }
                    else {
                        object_color = vec3(0., 0., 0.);
                    }
                    object_reflection = .25;
                    object_index = PLANE;
                }
                else if (t0 < t1) {
                    // Sphere 0.
                    M = rayO + rayD * t0;
                    object_normal = normalize(M - sphere_position_0);
                    object_color = sphere_color_0;
                    object_reflection = .5;
                    object_index = SPHERE_0;
                }
                else if (t1 < t0) {
                    // Sphere 1.
                    M = rayO + rayD * t1;
                    object_normal = normalize(M - sphere_position_1);
                    object_color = sphere_color_1;
                    object_reflection = .5;
                    object_index = SPHERE_1;
                }
                else {
                    break;
                }
                
                N = object_normal;
                toL = normalize(light_position - M);
                toO = normalize(O - M);
                
                // Shadow of the spheres on the plane.
                if (object_index == PLANE) {
                    t0 = intersect_sphere(M + N * .0001, toL, 
                                        sphere_position_0, sphere_radius_0);
                    t1 = intersect_sphere(M + N * .0001, toL, 
                                        sphere_position_1, sphere_radius_1);
                    if (min(t0, t1) < INFINITY) {
                        break;
                    }
                }
                
                col_ray = vec3(ambient, ambient, ambient);
                col_ray += light_intensity * max(dot(N, toL), 0.) * object_color;
                col_ray += light_specular.x * light_color * 
                    pow(max(dot(N, normalize(toL + toO)), 0.), light_specular.y);
                
                /* end trace_ray */
                
                rayO = M + N * .0001;
                rayD = normalize(rayD - 2. * dot(rayD, N) * N);
                col += reflection * col_ray;
                reflection *= object_reflection;
                
                depth++;
            }
            
            if (col==0 && v_position.y > 0.0) { // the biggest hack
               discard;
               }
            return clamp(col, 0., 1.);
        }
        
        void main() {
            vec2 pos = v_position;
            gl_FragColor = vec4(run(pos.x*u_aspect_ratio, pos.y, u_time), 1.);
        }
        """

    def __init__(self, **kwargs):
        scene.Visual.__init__(self, **kwargs)
        
        self.program = ModularProgram(self.vertex, self.fragment)
        self._vbo = gloo.VertexBuffer(np.array([(0., 0.), (0., 1.),
                                      (1., 0.), (1., 1.)], np.float32))
        self._timer = app.Timer('auto', connect=self.on_timer)
        self._t = 0.0
    
    def on_timer(self, event):
        self._t = event.elapsed

    def draw(self, event):
        # Set transform
        xform = event.render_transform.shader_map()
        self.program.vert['transform'] = xform
        self.program.prepare()  
        
        self.program['a_position'] = self._vbo
        self.program['u_time'] = self._t
        self.program['sphere_position_0'] = (+.75, .1, 2.0 + 1.0 * math.cos(4*self._t))
        self.program['sphere_position_1'] = (-.75, .1, 2.0 - 1.0 * math.cos(4*self._t))
        self.program['u_aspect_ratio'] = 1.0
        
        self.program['sphere_radius_0'] = .6
        self.program['sphere_color_0'] = (0., 0., 1.)
        
        self.program['sphere_radius_1'] = .6
        self.program['sphere_color_1'] = (.5, .223, .5)

        self.program['plane_position'] = (0., -.5, 0.)
        self.program['plane_normal'] = (0., 1., 0.)
        
        self.program['light_intensity'] = 1.
        self.program['light_specular'] = (1., 50.)
        self.program['light_position'] = (5., 5., -10.)
        self.program['light_color'] = (1., 1., 1.)
        self.program['ambient'] = .05
        self.program['O'] = (0., 0., -1.)
        
        self.program.draw('triangle_strip')
