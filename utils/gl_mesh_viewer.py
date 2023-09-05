from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image
import numpy as np
import glfw
import glfw.GLFW


class GL_VAO:

  def __init__(self) -> None:
    self.vao = glGenVertexArrays(1)

  def bind(self):
    glBindVertexArray(self.vao)

  def unbind(self):
    glBindVertexArray(0)


class GL_Buffer:

  def __init__(self, data: np.ndarray, target, draw_type) -> None:
    self.target = target
    self.buffer = glGenBuffers(1)
    self.draw_type = draw_type

    self.update_buffer_data(data)

  def bind(self):
    glBindBuffer(self.target, self.buffer)

  def unbind(self):
    glBindBuffer(self.target, 0)

  def update_buffer_data(self, data: np.ndarray):
    self.bind()
    glBufferData(self.target, data.nbytes, data, self.draw_type)
    self.unbind()


class GL_Texture2D:

  def __init__(self, filepath: str, img_type) -> None:
    self.texture = glGenTextures(1)
    self.bind()

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    image = Image.open(filepath)
    image = image.convert('RGBA')
    img_data = np.array(list(image.getdata()), np.uint8)
    image_size = image.size

    image_width = image_size[0]
    image_height = image_size[1]

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, img_type, image_width, image_height, 0,
                 img_type, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    self.unbind()

  def bind(self):
    glBindTexture(GL_TEXTURE_2D, self.texture)

  def unbind(self):
    glBindTexture(GL_TEXTURE_2D, 0)


class Mesh2dShaderProgram:

  def __init__(self, tex_channel=0) -> None:
    self.tex_channel = tex_channel
    self.program = self.mesh2d_shader_program()
    self.bind()
    self.pos_attri = glGetAttribLocation(self.program, "inPos")
    self.uv_attri = glGetAttribLocation(self.program, "inUVs")
    glUniform1i(glGetUniformLocation(self.program, "imageTexture"),
                self.tex_channel)
    glUniform1i(glGetUniformLocation(self.program, "wireframe"), False)
    self.wireframe = False
    self.unbind()

  def set_wireframe_mode(self,
                         wireframe: bool,
                         color: tuple = (0.0, 0.0, 0.0, 1.0)):
    self.bind()
    self.wireframe = wireframe
    if wireframe:
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glUniform1i(glGetUniformLocation(self.program, "wireframe"), wireframe)
    glUniform4f(glGetUniformLocation(self.program, "wireframe_color"), color[0],
                color[1], color[2], color[3])
    self.unbind()

  def bind(self):
    glUseProgram(self.program)

  def unbind(self):
    glUseProgram(0)

  def bind_buffer(self, vert_buffer: GL_Buffer, uv_buffer: GL_Buffer):
    vert_buffer.bind()
    glEnableVertexAttribArray(self.pos_attri)
    glVertexAttribPointer(self.pos_attri, 2, GL_FLOAT, GL_FALSE, 0,
                          ctypes.c_void_p(0))
    vert_buffer.unbind()
    uv_buffer.bind()
    glEnableVertexAttribArray(self.uv_attri)
    glVertexAttribPointer(self.uv_attri, 2, GL_FLOAT, GL_FALSE, 0,
                          ctypes.c_void_p(0))
    uv_buffer.unbind()

  def bind_texture(self, texture: GL_Texture2D):
    glActiveTexture(GL_TEXTURE0)
    texture.bind()

  def mesh2d_shader_program(self):
    vertex_shader = """#version 330 core
        in vec2 inPos;
        in vec2 inUVs;
        // in vec4 colors;
        out vec2 outUVs;

        void main()
        {
            gl_Position = vec4(inPos*2.0-1.0, 0.0, 1.0);
            outUVs = vec2(inUVs[0], 1.0 - inUVs[1]);
        }
      """
    fragment_shader = """#version 330 core
        in vec2 outUVs;
        out vec4 color;

        uniform sampler2D imageTexture;
        uniform bool wireframe;
        uniform vec4 wireframe_color;

        void main()
        {
            if (wireframe){
              color = wireframe_color;
            }
            else{
              color = texture(imageTexture, outUVs);
            }
        }
      """
    return shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))


class GL_Mesh2D:

  def __init__(self, verts, uvs, faces, dynamic_verts=True) -> None:
    self.vert_draw_type = GL_DYNAMIC_DRAW if dynamic_verts else GL_STATIC_DRAW
    self.vao = GL_VAO()
    self.vao.bind()
    self.n_faces = faces.size // 3
    self.vert_buffer = GL_Buffer(verts.flatten(),
                                 target=GL_ARRAY_BUFFER,
                                 draw_type=self.vert_draw_type)
    self.uv_buffer = GL_Buffer(uvs.flatten(),
                               target=GL_ARRAY_BUFFER,
                               draw_type=GL_STATIC_DRAW)
    self.face_buffer = GL_Buffer(faces.flatten(),
                                 target=GL_ELEMENT_ARRAY_BUFFER,
                                 draw_type=GL_STATIC_DRAW)

  def update_verts(self, verts: np.ndarray):
    assert self.vert_draw_type == GL_DYNAMIC_DRAW
    self.vert_buffer.update_buffer_data(verts.flatten())

  def bind(self):
    self.vao.bind()
    self.face_buffer.bind()

  def unbind(self):
    self.face_buffer.unbind()
    self.vao.unbind()

  def draw(self):
    glDrawElements(GL_TRIANGLES, self.n_faces * 3, GL_UNSIGNED_INT, None)


# ! maybe find a method to set which GPU to use
class OpenGLMeshRenderer2D:

  def __init__(self, title: str, res) -> None:
    if not glfw.init():
      return -1

    self.window = glfw.create_window(res[0], res[1], title, None, None)
    self.res = res
    self.fps_count = 0
    self.frame_count = 0
    self.prev_time = glfw.get_time()
    self.dragging = False
    self.dragging_right = False
    self.start_drag_pos = np.zeros(dtype=np.float32, shape=2)
    self.start_drag_pos_right = np.zeros(dtype=np.float32, shape=2)
    glfw.window_hint(glfw.GLFW.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.GLFW.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.GLFW.GLFW_OPENGL_PROFILE,
                     glfw.GLFW.GLFW_OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.GLFW.GLFW_SAMPLES, 4)
    #glEnable(GL_MULTISAMPLE)

    if not self.window:
      glfw.terminate()
      return -1

    # Make the window's context current
    glfw.make_context_current(self.window)
    glfw.swap_interval(1)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_LINE_SMOOTH)

    glfw.set_input_mode(self.window, glfw.STICKY_KEYS, GL_TRUE)

    self.key_input_callback = []
    self.mouse_input_callback = []
    self.cursor_move_callback = []
    self.wireframe_color = (0.0, 0.0, 0.0, 1.0)
    self.wireframe_mode = False

    def key_event(window, key, scancode, action, mods):
      for func in self.key_input_callback:
        func(key, scancode, action, mods)

    def mouse_input_event(window, button, action, mods):
      for func in self.mouse_input_callback:
        func(button, action, mods)

    def cursor_move_event(window, xpos, ypos):
      for func in self.cursor_move_callback:
        func(float(xpos), float(ypos))

    def set_drag(button, action, mods):
      p = self.get_cursor_pos()
      if p[1] > 0.9:
        return
      if action == glfw.PRESS and button == glfw.MOUSE_BUTTON_LEFT:
        self.start_drag_pos = self.get_cursor_pos()
        self.dragging = True
      if action == glfw.RELEASE and button == glfw.MOUSE_BUTTON_LEFT:
        self.dragging = False
      if action == glfw.PRESS and button == glfw.MOUSE_BUTTON_RIGHT:
        self.start_drag_pos_right = self.get_cursor_pos()
        self.dragging_right = True
      if action == glfw.RELEASE and button == glfw.MOUSE_BUTTON_RIGHT:
        self.dragging_right = False

    def set_view_mode(key, scancode, action, mods):
      if action == glfw.PRESS and key == glfw.KEY_F:
        self.wireframe_mode = not self.wireframe_mode
        if self.wireframe_mode:
          self.set_wireframe_mode(True, color=self.wireframe_color)
        else:
          self.set_wireframe_mode(False)

    glfw.set_key_callback(self.window, key_event)
    glfw.set_mouse_button_callback(self.window, mouse_input_event)
    glfw.set_cursor_pos_callback(self.window, cursor_move_event)

    self.add_mouse_callback(set_drag)
    self.add_key_callback(set_view_mode)

    self.running = True
    self.data_set = False

    glfw.set_time(0)

    self.movie_track_mode = False
    self.movie_track_frames = -1
    self.fps_track = False

  def set_fps_record(self, totalframes):
    self.fps_record = np.zeros(dtype=np.float32, shape=totalframes)
    self.fps_totalframes = totalframes
    self.fps_track = True
    self.prev_fps_time = glfw.get_time()

  def set_movie_track(self, max_frames, frame_per_track=1):
    if self.frame_count > 0:
      assert False, 'Error! Movie track mode should be set at the beginning'
    self.movie_track_mode = True
    self.movie_track_frames = max_frames
    self.frame_per_track = frame_per_track

  def get_time(self):
    return glfw.get_time()

  def add_key_callback(self, func):
    self.key_input_callback.append(func)

  def add_mouse_callback(self, func):
    self.mouse_input_callback.append(func)

  def add_cursor_move_callback(self, func):
    self.cursor_move_callback.append(func)

  def set_mesh(self, verts: np.ndarray, uvs: np.ndarray, faces: np.ndarray,
               texpath: str):
    self.verts = verts
    self.uvs = uvs
    self.faces = faces
    self.shaderProgram = Mesh2dShaderProgram()
    self.shaderProgram.bind()

    self.vao = GL_VAO()
    self.vao.bind()

    self.gl_mesh = GL_Mesh2D(verts, uvs, faces, dynamic_verts=True)
    self.texture = GL_Texture2D(texpath, GL_RGBA)

    self.shaderProgram.bind_buffer(self.gl_mesh.vert_buffer,
                                   self.gl_mesh.uv_buffer)
    self.shaderProgram.bind_texture(self.texture)

    self.vao.unbind()
    self.shaderProgram.unbind()
    self.texture.unbind()

    self.data_set = True

  def set_wireframe_mode(self,
                         wireframe: bool,
                         color: tuple = (0.1, 0.4, 1.0, 0.8)):
    self.shaderProgram.set_wireframe_mode(wireframe, color)

  def draw_lines(self,
                 verts: np.ndarray,
                 edges: np.ndarray,
                 line_width=5.0,
                 line_color=(0.8, 0.0, 0.7)):
    edges = edges.reshape(-1, 2)
    glColor3f(line_color[0], line_color[1], line_color[2])
    glLineWidth(line_width)
    glBegin(GL_LINES)
    for edge in edges:
      glVertex2f(verts[edge[0], 0] * 2.0 - 1.0, verts[edge[0], 1] * 2.0 - 1.0)
      glVertex2f(verts[edge[1], 0] * 2.0 - 1.0, verts[edge[1], 1] * 2.0 - 1.0)
    glEnd()

  def update_mesh(self, verts: np.ndarray):
    self.verts = verts
    self.gl_mesh.bind()
    self.gl_mesh.update_verts(verts)
    self.gl_mesh.unbind()

  def pre_update(self):
    self.running = not glfw.window_should_close(self.window)
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

  def data_render(self):
    assert self.data_set
    glLineWidth(1.0)
    self.shaderProgram.bind()
    self.texture.bind()
    self.gl_mesh.bind()
    self.gl_mesh.draw()
    self.gl_mesh.unbind()
    self.shaderProgram.unbind()

  def show(self):
    if self.movie_track_mode and self.frame_count < self.movie_track_frames and self.frame_count > 0:
      if self.frame_count % self.frame_per_track == 0:
        idx = self.frame_count // self.frame_per_track
        self.get_screneshot(f'track/{idx:05d}.png')
    if self.movie_track_mode and self.frame_count == self.movie_track_frames:
      print("movie track complete")
    self.fps_count = int(1.0 / (glfw.get_time() - self.prev_time))
    if self.fps_track and self.frame_count < self.fps_totalframes:
      self.fps_record[self.frame_count] = self.fps_count
    glfw.swap_buffers(self.window)
    self.prev_time = glfw.get_time()
    self.frame_count += 1

  def get_cursor_pos(self) -> np.ndarray:
    cps = np.array(glfw.get_cursor_pos(self.window))
    cps[0] = cps[0] / self.res[0]
    cps[1] = 1.0 - cps[1] / self.res[1]
    #cps = (cps + 1.0) / 2.0
    return cps

  def terminate(self):
    if self.fps_track:
      print(
          f"avg fps is {np.sum(self.fps_record[100:]) / (self.fps_totalframes - 100):.3f}"
      )
    glfw.terminate()

  def _key_input(self):
    pass

  def get_mesh_png(self, filepath, face_color: np.ndarray):
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(self.verts[:, 0],
                        self.verts[:, 1],
                        self.faces.reshape(self.faces.size // 3, 3),
                        facecolors=face_color,
                        shading='flat')
    fig1.colorbar(tpc)
    plt.savefig(filepath)
    # ax1.set_title('Distortion of triangle elements')

    plt.show()

  def get_screneshot(self, filepath):
    pixels = np.zeros(dtype=np.uint8, shape=3 * self.res[0] * self.res[1])
    glReadPixels(0, 0, self.res[0], self.res[1], GL_RGB, GL_UNSIGNED_BYTE,
                 pixels)
    img = Image.fromarray(pixels.reshape(
        (self.res[0], self.res[1],
         3))).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
    img.save(filepath)
