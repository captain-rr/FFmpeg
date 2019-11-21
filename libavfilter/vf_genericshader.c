#include <string.h>
#include <float.h>
#include "internal.h"

#include "avfilter.h"
#include "formats.h"
#include "libavutil/common.h"
#include "libavutil/file.h"
#include "libavutil/eval.h"
#include "libavutil/avstring.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"
#include "libavutil/timestamp.h"
#include "drawutils.h"
#include "video.h"

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include "glew.c"
#endif

#include <GLFW/glfw3.h>

enum var_name {
    VAR_MAIN_W,    VAR_MW,
    VAR_MAIN_H,    VAR_MH,
    VAR_POWER,
    VAR_N,
    VAR_POS,
    VAR_T,
    VAR_VARS_NB
};

typedef struct GenericShaderContext {
    const AVClass *class;

    GLuint        program;
    GLuint        frame_tex;
    GLFWwindow    *window;
    GLuint        pos_buf;
    char          *shader_style;
    int           frame_idx;

    int is_color; // for vintage filter

    float dropSize; // for matrix filter

    char *vs_textfile;
    uint8_t *vs_text;
    char *fs_textfile;
    uint8_t *fs_text;

    int eval_mode;              ///< EvalMode
    double var_values[VAR_VARS_NB];
    float power;
    char *power_expr;
    AVExpr *power_pexpr;
} GenericShaderContext;

static const char *const var_names[] = {
    "main_w",    "W", ///< width  of the main    video
    "main_h",    "H", ///< height of the main    video
    "power",
    "n",            ///< number of frame
    "pos",          ///< position in the file
    "t",            ///< timestamp expressed in seconds
    NULL
};

enum EvalMode {
    EVAL_MODE_INIT,
    EVAL_MODE_FRAME,
    EVAL_MODE_NB
};

enum IsColorMode {
    IS_COLOR_MODE_TRUE,
    IS_COLOR_MODE_FALSE,
    IS_COLOR_MODE_NB
};

static const float position[12] = {
  -1.0f, -1.0f,  //A
  1.0f, -1.0f,  // B
  -1.0f, 1.0f,  //C

  -1.0f, 1.0f, //C
  1.0f, -1.0f,  //B
  1.0f, 1.0f}; //D

static const GLchar *v_shader_source =
  "attribute vec2 position;\n"
  "varying vec2 texCoord;\n"
  "void main(void) {\n"
  "  gl_Position = vec4(position, 0, 1);\n"
  "  texCoord = position* 0.5 + 0.5;\n"
  "}\n";

static const GLchar *f_shader_source =
  "varying vec2 texCoord;\n"
  "uniform sampler2D tex;\n"
  "void main() {\n"
  "  gl_FragColor = texture2D(tex, texCoord);\n"
  "}\n";

// Thanks to luluco250 - https://www.shadertoy.com/view/4t2fRz
static const GLchar *f_vintage_shader_source =
"// 0: Addition, 1: Screen, 2: Overlay, 3: Soft Light, 4: Lighten-Only\n"
"#define BLEND_MODE 0\n"
"#define SPEED 2.0\n"
"#define INTENSITY 0.075\n"
"// What gray level noise should tend to.\n"
"#define MEAN 0.0\n"
"// Controls the contrast/variance of noise.\n"
"#define VARIANCE 0.5\n"

"vec3 channel_mix(vec3 a, vec3 b, vec3 w) {\n"
"    return vec3(mix(a.r, b.r, w.r), mix(a.g, b.g, w.g), mix(a.b, b.b, w.b));\n"
"}\n"

"float gaussian(float z, float u, float o) {\n"
"    return (1.0 / (o * sqrt(2.0 * 3.1415))) * exp(-(((z - u) * (z - u)) / (2.0 * (o * o))));\n"
"}\n"

"vec3 madd(vec3 a, vec3 b, float w) {\n"
"    return a + a * b * w;\n"
"}\n"

"vec3 screen(vec3 a, vec3 b, float w) {\n"
"    return mix(a, vec3(1.0) - (vec3(1.0) - a) * (vec3(1.0) - b), w);\n"
"}\n"

"vec3 overlay(vec3 a, vec3 b, float w) {\n"
"    return mix(a, channel_mix(\n"
"        2.0 * a * b,\n"
"        vec3(1.0) - 2.0 * (vec3(1.0) - a) * (vec3(1.0) - b),\n"
"        step(vec3(0.5), a)\n"
"    ), w);\n"
"}\n"

"vec3 soft_light(vec3 a, vec3 b, float w) {\n"
"    return mix(a, pow(a, pow(vec3(2.0), 2.0 * (vec3(0.5) - b))), w);\n"
"}\n"
"varying vec2 texCoord;\n"
"uniform sampler2D text;\n"
"uniform float power;\n"
"uniform float time;\n"
"uniform bool isColor;\n"

"void main() {\n"
"    vec2 uv = texCoord;\n"
"    vec4 color = texture2D(text, uv);\n"
"    vec4 originalColor = texture2D(text, uv);\n"
"    float t = time * float(SPEED);\n"
"    float seed = dot(uv, vec2(12.9898, 78.233));\n"
"    float noise = fract(sin(seed) * 43758.5453 + t);\n"
"    noise = gaussian(noise, float(MEAN), float(VARIANCE) * float(VARIANCE));\n"
"    float w = float(INTENSITY);\n"
"    vec3 grain = vec3(noise) * (1.0 - color.rgb);\n"
"    color.rgb += grain * w;\n"
"    if(isColor){\n"
"        color.r = color.r * 1.1 + 0.1 ;\n"
"        color.b = color.b * 0.8 + 0.2 ;\n"
"        color.g = color.g * color.b * 1.8 - 0.08;\n"
"    }\n"
"    gl_FragColor = mix(originalColor,color,power);\n"
"}\n";

static const GLchar *f_shockwave_shader_source =
"#define ZERO vec2(0.0,0.0)\n"
"uniform float power;\n"
"uniform float u_time;\n"

"varying vec2 texCoord;\n"
"uniform sampler2D tex;\n"
"uniform float time;\n"
"vec3 invert(vec3 rgb)\n"
"{\n"
"    return vec3(1.0-rgb.r,1.0-rgb.g,1.0-rgb.b);\n"
"}\n"

"void handleRadius(out float radius,vec2 uvc,float progress2,float uvcd, out vec3 color, bool isMax){\n"
"    float pMul = 1.440;\n"
"    radius *= (cos(uvc.x * 33.752 + pMul * progress2) * 0.024 + (1.0 - uvcd*0.040)) ;\n"
"    radius *= (sin(uvc.y * 33.752 + pMul * progress2) * 0.008 + (1.0 - uvcd*0.040)) ;\n"

"    bool inside = uvcd > radius ? true : false;\n"
"    float ringWidth = inside ? 0.029 : 0.221;\n"
"    float disFromRing = abs(uvcd - radius);\n"
"    if(uvcd < radius){\n"
"    color.rgb = invert(color.rgb);\n"
"    }\n"

"    if(isMax ? (disFromRing < ringWidth) : (disFromRing > ringWidth)){\n"
"        float lightPower = inside ? 27.312 : 2.856;\n"
"        color.rgb += vec3(1.,1.,1.) * lightPower * (ringWidth - disFromRing);\n"
"    }\n"
"}\n"

"vec2 SineWave( vec2 p,float tx, float ty )\n"
"    {\n"
"    // convert Vertex position <-1,+1> to texture coordinate <0,1> and some shrinking so the effect dont overlap screen\n"
"    // wave distortion\n"
"    float x = sin( 25.0*p.y + 30.0*p.x + 6.28*tx) * 0.01;\n"
"    float y = sin( 25.0*p.y + 30.0*p.x + 6.28*ty) * 0.01;\n"
"    return vec2(p.x+x, p.y+y);\n"
"    }\n"

"void main() {\n"
"    vec2 uv = texCoord;\n"
"    float progress2 = (power / 2.0 - 0.1) * 1.2;\n"
"    \n"
"    vec2 disorted_uv = mix(texCoord, SineWave(texCoord, time, time), power);\n"
"    \n"
"    vec3 color = texture2D(tex, disorted_uv).rgb;\n"
"    vec2 uvc = vec2(0.5,0.5) - uv;\n"
"    \n"
"    float minRadius = progress2  - 0.5;\n"
"    float maxRadius = progress2 * 2.0;\n"
"    float uvcd = distance(ZERO, uvc);\n"
"    \n"
"  	handleRadius(maxRadius,uvc,progress2,uvcd,color,true);\n"
"    //handleRadius(minRadius,uvc,progress2,uvcd,color,true);\n"

"    gl_FragColor = vec4(color,1.0);\n"
"}\n";

// thanks to raja https://www.shadertoy.com/view/lsXSDn
static const GLchar *f_matrix_shader_source =
"#define RAIN_SPEED 0.002 // Speed of rain droplets\n"
"#define DROP_SIZE  3.0  // Higher value lowers, the size of individual droplets\n"

"float rand(vec2 co){\n"
"    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);\n"
"}\n"

"float rchar(vec2 outer, vec2 inner, float globalTime) {\n"
"    //return float(rand(floor(inner * 2.0) + outer) > 0.9);\n"

"    vec2 seed = floor(inner * 4.0) + outer.y;\n"
"    if (rand(vec2(outer.y, 23.0)) > 0.98) {\n"
"        seed += floor((globalTime + rand(vec2(outer.y, 49.0))) * 3.0);\n"
"    }\n"

"    return float(rand(seed) > 0.5);\n"
"}\n"

"varying vec2 texCoord;\n"
"uniform sampler2D tex;\n"
"uniform float power;\n"
"uniform float time;\n"
"uniform float dropSize;\n"

"void main() {\n"
"    vec4 originalColor = texture2D(tex, texCoord);\n"
"    float globalTime = time * RAIN_SPEED + 34284.0;\n"
"    vec2 position = - texCoord;\n"
"    vec2 res = vec2(700,300);\n"
"    position.x /= res.x / res.y;\n"
"    float scaledown = dropSize;\n"
"    float rx = texCoord.x * res.x / (40.0 * scaledown);\n"
"    float mx = 40.0*scaledown*fract(position.x * 30.0 * scaledown);\n"
"    vec4 result;\n"

"    if (mx > 12.0 * scaledown) {\n"
"        result = vec4(0.0);\n"
"    } else\n"
"    {\n"
"        float x = floor(rx);\n"
"        float r1x = floor(texCoord.x * res.x / (15.0));\n"


"        float ry = position.y*600.0 + rand(vec2(x, x * 3.0)) * 100000.0 + globalTime* rand(vec2(r1x, 23.0)) * 120.0;\n"
"        float my = mod(ry, 15.0);\n"
"        if (my > 12.0 * scaledown) {\n"
"            result = vec4(0.0);\n"
"        } else {\n"

"            float y = floor(ry / 15.0);\n"

"            float b = rchar(vec2(rx, floor((ry) / 15.0)), vec2(mx, my) / 12.0, globalTime);\n"
"            float col = max(mod(-y, 24.0) - 4.0, 0.0) / 20.0;\n"
"            vec3 c = col < 0.8 ? vec3(0.0, col / 0.8, 0.0) : mix(vec3(0.0, 1.0, 0.0), vec3(1.0), (col - 0.8) / 0.2);\n"

"            result = vec4(c * b, 1.0)  ;\n"
"        }\n"
"    }\n"

"    position.x += 0.05;\n"

"    scaledown = dropSize;\n"
"    rx = texCoord.x * res.x / (40.0 * scaledown);\n"
"    mx = 40.0*scaledown*fract(position.x * 30.0 * scaledown);\n"

"    if (mx > 12.0 * scaledown) {\n"
"        result += vec4(0.0);\n"
"    } else\n"
"    {\n"
"        float x = floor(rx);\n"
"        float r1x = floor(texCoord.x * res.x / (12.0));\n"


"        float ry = position.y*700.0 + rand(vec2(x, x * 3.0)) * 100000.0 + globalTime* rand(vec2(r1x, 23.0)) * 120.0;\n"
"        float my = mod(ry, 15.0);\n"
"        if (my > 12.0 * scaledown) {\n"
"            result += vec4(0.0);\n"
"        } else {\n"

"            float y = floor(ry / 15.0);\n"

"            float b = rchar(vec2(rx, floor((ry) / 15.0)), vec2(mx, my) / 12.0, globalTime);\n"
"            float col = max(mod(-y, 24.0) - 4.0, 0.0) / 20.0;\n"
"            vec3 c = col < 0.8 ? vec3(0.0, col / 0.8, 0.0) : mix(vec3(0.0, 1.0, 0.0), vec3(1.0), (col - 0.8) / 0.2);\n"

"            result += vec4(c * b, 1.0)  ;\n"
"        }\n"
"    }\n"

"    result = 0.5 * result * length(originalColor) + 0.8 * originalColor;\n"
"    if(result.b < 0.5)\n"
"    result.b = result.g * 0.5 ;\n"
"    gl_FragColor = mix(originalColor,result,power);\n"
"}\n";

#define PIXEL_FORMAT GL_RGB
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
#define MAIN    0

static inline float normalize_power(double d)
{
    if (isnan(d))
        return FLT_MAX;
    return (float)d;
}

static void eval_expr(AVFilterContext *ctx)
{
    GenericShaderContext *s = ctx->priv;
    av_log(ctx, AV_LOG_VERBOSE, "eval_expr\n");

    s->var_values[VAR_POWER] = av_expr_eval(s->power_pexpr, s->var_values, NULL);
    s->power = normalize_power(s->var_values[VAR_POWER]);
    av_log(ctx, AV_LOG_VERBOSE, "eval_expr end\n");
}

static int set_expr(AVExpr **pexpr, const char *expr, const char *option, void *log_ctx)
{
    int ret;
    AVExpr *old = NULL;

    av_log(log_ctx, AV_LOG_VERBOSE, "set_expr expr:'%s' option:%s\n", expr, option);

    if (*pexpr)
        old = *pexpr;
    ret = av_expr_parse(pexpr, expr, var_names,
                        NULL, NULL, NULL, NULL, 0, log_ctx);
    if (ret < 0) {
        av_log(log_ctx, AV_LOG_ERROR,
               "Error when evaluating the expression '%s' for %s\n",
               expr, option);
        *pexpr = old;
        return ret;
    }

    av_expr_free(old);
    av_log(log_ctx, AV_LOG_VERBOSE, "set_expr end\n");
    return 0;
}

static int process_command(AVFilterContext *ctx, const char *cmd, const char *args,
                           char *res, int res_len, int flags)
{
    int ret;
    GenericShaderContext *s = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "process_command cmd:%s args:%s\n",
           cmd, args);

    if (strcmp(cmd, "power") == 0)
        ret = set_expr(&s->power_pexpr, args, cmd, ctx);
    else
        ret = AVERROR(ENOSYS);

    if (ret < 0)
        return ret;

    if (s->eval_mode == EVAL_MODE_INIT) {
        eval_expr(ctx);
        av_log(ctx, AV_LOG_VERBOSE, "pow:%f powi:%f\n",
               s->var_values[VAR_POWER], s->power);
    }
    av_log(ctx, AV_LOG_VERBOSE, "process_command end\n");
    return ret;
}

static int load_textfile(AVFilterContext *ctx, char *textfile, uint8_t **text)
{
    int err;
    uint8_t *textbuf;
    uint8_t *tmp;
    size_t textbuf_size;

    if ((err = av_file_map(textfile, &textbuf, &textbuf_size, 0, ctx)) < 0) {
        av_log(ctx, AV_LOG_ERROR,
               "The text file '%s' could not be read or is empty\n",
               textfile);
        return err;
    }
    av_log(ctx, AV_LOG_VERBOSE,
           "load_textfile '%s' 2\n",
           textfile);

    if (textbuf_size > SIZE_MAX - 1 ||
        !(tmp = av_realloc(*text, textbuf_size + 1))) {
        av_file_unmap(textbuf, textbuf_size);
        av_log(ctx, AV_LOG_ERROR,
               "The text file '%s' created buffer size issue\n",
               textfile);
        return AVERROR(ENOMEM);
    }
    av_log(ctx, AV_LOG_VERBOSE,
           "load_textfile '%s' 3\n",
           textfile);
    *text = tmp;
    memcpy(*text, textbuf, textbuf_size);
    av_log(ctx, AV_LOG_VERBOSE,
           "load_textfile '%s' 4\n",
           textfile);
    *text[textbuf_size] = 0;
    av_file_unmap(textbuf, textbuf_size);
    av_log(ctx, AV_LOG_VERBOSE,
           "load_textfile '%s' 5\n",
           textfile);

    return 0;
}

static GLuint build_shader(AVFilterContext *ctx, const GLchar *shader_source, GLenum type) {
    GLint status;
    GLuint shader = glCreateShader(type);
    av_log(ctx, AV_LOG_VERBOSE, "build_shader\n");
    if (!shader || !glIsShader(shader)) {
        return 0;
    }
    glShaderSource(shader, 1, &shader_source, 0);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    av_log(ctx, AV_LOG_VERBOSE, "build_shader end %d\n", status);
    return status == GL_TRUE ? shader : 0;
}

static void vbo_setup(GenericShaderContext *gs, AVFilterContext *log_ctx) {
  GLint loc;
  av_log(log_ctx, AV_LOG_VERBOSE, "vbo_setup\n");
  glGenBuffers(1, &gs->pos_buf);
  glBindBuffer(GL_ARRAY_BUFFER, gs->pos_buf);
  glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);

  loc = glGetAttribLocation(gs->program, "position");
  glEnableVertexAttribArray(loc);
  glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
  av_log(log_ctx, AV_LOG_VERBOSE, "vbo_setup end\n");
}

static void tex_setup(AVFilterLink *inlink, AVFilterContext *log_ctx) {
    AVFilterContext     *ctx = inlink->dst;
    GenericShaderContext *gs = ctx->priv;
    av_log(log_ctx, AV_LOG_VERBOSE, "tex_setup\n");

    glGenTextures(1, &gs->frame_tex);
    glActiveTexture(GL_TEXTURE0);

    glBindTexture(GL_TEXTURE_2D, gs->frame_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, NULL);

    glUniform1i(glGetUniformLocation(gs->program, "tex"), 0);
    if (!strcmp(gs->shader_style, "matrix")){
        glUniform1f(glGetUniformLocation(gs->program, "power"), 0);
        glUniform1f(glGetUniformLocation(gs->program, "time"), 0);
        glUniform1f(glGetUniformLocation(gs->program, "dropSize"), 0);
    } else if (!strcmp(gs->shader_style, "shockwave")){
        glUniform1f(glGetUniformLocation(gs->program, "power"), 0);
        glUniform1f(glGetUniformLocation(gs->program, "time"), 0);
    } else if (!strcmp(gs->shader_style, "vintage")){
        glUniform1f(glGetUniformLocation(gs->program, "power"), 0);
        glUniform1f(glGetUniformLocation(gs->program, "time"), 0);
    }
    av_log(log_ctx, AV_LOG_VERBOSE, "tex_setup end\n");
}

static int build_program(AVFilterContext *ctx) {
    GLint status;
    GLuint v_shader, f_shader;
    GLchar* shader;
    GenericShaderContext *gs = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "build_program %s\n", gs->shader_style);

    if (gs->vs_text){
        av_log(ctx, AV_LOG_VERBOSE, "build_program vs_from_text ||%s||\n", gs->vs_text);
        v_shader = build_shader(ctx, (GLchar*)gs->vs_text, GL_VERTEX_SHADER);
    } else {
        v_shader = build_shader(ctx, v_shader_source, GL_VERTEX_SHADER);
    }

    if (gs->fs_text){
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_s %d ||%s||\n", v_shader, gs->fs_text);
        f_shader = build_shader(ctx, (GLchar*)gs->fs_text, GL_FRAGMENT_SHADER);
    } else if (!strcmp(gs->shader_style, "matrix")){
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_1 %d\n", v_shader);
        f_shader = build_shader(ctx, f_matrix_shader_source, GL_FRAGMENT_SHADER);
    } else if (!strcmp(gs->shader_style, "shockwave")) {
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_2 %d\n", v_shader);
        f_shader = build_shader(ctx, f_shockwave_shader_source, GL_FRAGMENT_SHADER);
    } else if (!strcmp(gs->shader_style, "vintage")) {
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_3 %d\n", v_shader);
        f_shader = build_shader(ctx, f_vintage_shader_source, GL_FRAGMENT_SHADER);
    } else {
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_0 %d\n", v_shader);
        f_shader = build_shader(ctx, f_shader_source, GL_FRAGMENT_SHADER);
    }

    av_log(ctx, AV_LOG_VERBOSE, "build_program_3 %d\n", f_shader);

    if (!(v_shader && f_shader)) {
        av_log(ctx, AV_LOG_VERBOSE, "build_program shader build fail\n");
        return -1;
    }

    gs->program = glCreateProgram();
    av_log(ctx, AV_LOG_VERBOSE, "build_program_4 %d\n", gs->program);
    glAttachShader(gs->program, v_shader);
    glAttachShader(gs->program, f_shader);
    glLinkProgram(gs->program);
    av_log(ctx, AV_LOG_VERBOSE, "build_program_5\n");

    glGetProgramiv(gs->program, GL_LINK_STATUS, &status);
    av_log(ctx, AV_LOG_VERBOSE, "build_program end %d\n", status);
    return status == GL_TRUE ? 0 : -1;
}

static av_cold int init(AVFilterContext *ctx) {
    int err;
    GenericShaderContext *gs = ctx->priv;
    av_log(ctx, AV_LOG_VERBOSE, "init\n");
    if (gs->vs_textfile) {
        av_log(ctx, AV_LOG_VERBOSE, "attempt load text file for vertex shader '%s'\n", gs->vs_textfile);
        if ((err = load_textfile(ctx, gs->vs_textfile, &gs->vs_text)) < 0)
            return err;
    }
    if (gs->fs_textfile) {
        av_log(ctx, AV_LOG_VERBOSE, "attempt load text file for fragment shader '%s'\n", gs->fs_textfile);
        if ((err = load_textfile(ctx, gs->fs_textfile, &gs->fs_text)) < 0)
            return err;
    }
    if (!gs->shader_style) {
        av_log(ctx, AV_LOG_ERROR, "Empty output shader style string.\n");
        return AVERROR(EINVAL);
    }

    return glfwInit() ? 0 : -1;
}

static int config_props(AVFilterLink *inlink) {
  int ret;
  AVFilterContext     *ctx = inlink->dst;
  GenericShaderContext *gs = ctx->priv;

  av_log(ctx, AV_LOG_VERBOSE, "config_props\n");

  glfwWindowHint(GLFW_VISIBLE, 0);
  gs->window = glfwCreateWindow(inlink->w, inlink->h, "", NULL, NULL);
  glfwMakeContextCurrent(gs->window);

  #ifndef __APPLE__
  glewExperimental = GL_TRUE;
  glewInit();
  #endif

  glViewport(0, 0, inlink->w, inlink->h);

  if ((ret = build_program(ctx)) < 0) {
    return ret;
  }
  gs->var_values[VAR_MAIN_W] = gs->var_values[VAR_MW] = ctx->inputs[MAIN]->w;
  gs->var_values[VAR_MAIN_H] = gs->var_values[VAR_MH] = ctx->inputs[MAIN]->h;
  gs->var_values[VAR_POWER]  = NAN;
  gs->var_values[VAR_T]      = NAN;
  gs->var_values[VAR_POS]    = NAN;

  if ((ret = set_expr(&gs->power_pexpr,      gs->power_expr,      "power",      ctx)) < 0)
      return ret;

  if (gs->eval_mode == EVAL_MODE_INIT) {
      eval_expr(ctx);
      av_log(ctx, AV_LOG_INFO, "pow:%f powi:%f\n",
             gs->var_values[VAR_POWER], gs->power);
  }

  av_log(ctx, AV_LOG_VERBOSE,
         "main w:%d h:%d\n",
         ctx->inputs[MAIN]->w, ctx->inputs[MAIN]->h,
         av_get_pix_fmt_name(ctx->inputs[MAIN]->format));

  glUseProgram(gs->program);
  vbo_setup(gs, ctx);
  tex_setup(inlink, ctx);

  return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in) {
    AVFilterContext *ctx     = inlink->dst;
    AVFilterLink    *outlink = ctx->outputs[0];
    GenericShaderContext *gs = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "filter_frame\n");

    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    if (gs->eval_mode == EVAL_MODE_FRAME) {
      int64_t pos = in->pkt_pos;

      gs->var_values[VAR_N] = inlink->frame_count_out;
      gs->var_values[VAR_T] = in->pts == AV_NOPTS_VALUE ? NAN : in->pts * av_q2d(inlink->time_base);
      gs->var_values[VAR_POS] = pos == -1 ? NAN : pos;

      gs->var_values[VAR_MAIN_W] = gs->var_values[VAR_MW] = in->width;
      gs->var_values[VAR_MAIN_H] = gs->var_values[VAR_MH] = in->height;

      eval_expr(ctx);
      av_log(ctx, AV_LOG_VERBOSE, "filter_frame pow:%f powi:%f time:%f\n",
             gs->var_values[VAR_POWER], gs->power, gs->var_values[VAR_T]);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, in->data[0]);
    if (!strcmp(gs->shader_style, "matrix")){
        glUniform1fv(glGetUniformLocation(gs->program, "power"), 1, &gs->power);
        GLfloat time = (GLfloat)(gs->var_values[VAR_T] == NAN? gs->frame_idx * 330: gs->var_values[VAR_T] * 1000);
        GLfloat dropSize = (GLfloat)gs->dropSize;
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame matrix time:%f dropSize:%f\n", time, dropSize);
        glUniform1fv(glGetUniformLocation(gs->program, "time"), 1, &time);
        glUniform1fv(glGetUniformLocation(gs->program, "dropSize"), 1, &dropSize);
    } else if (!strcmp(gs->shader_style, "shockwave")){
        glUniform1fv(glGetUniformLocation(gs->program, "power"), 1, &gs->power);
        GLfloat time = (GLfloat)(gs->var_values[VAR_T] == NAN? gs->frame_idx * 1.6667: gs->var_values[VAR_T] * 5);
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame shockwave time:%f\n", time);
        glUniform1fv(glGetUniformLocation(gs->program, "time"), 1, &time);
    } else if (!strcmp(gs->shader_style, "vintage")){
        glUniform1fv(glGetUniformLocation(gs->program, "power"), 1, &gs->power);
        GLfloat time = (GLfloat)(gs->var_values[VAR_T] == NAN? gs->frame_idx * 330: gs->var_values[VAR_T] * 1000);
        GLint isColor = gs->is_color == IS_COLOR_MODE_TRUE? 1: 0;
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame vintage isColor:%d time:%f\n", isColor, time);
        glUniform1fv(glGetUniformLocation(gs->program, "time"), 1, &time);
        glUniform1iv(glGetUniformLocation(gs->program, "isColor"), 1, &isColor);
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glReadPixels(0, 0, outlink->w, outlink->h, PIXEL_FORMAT, GL_UNSIGNED_BYTE, (GLvoid *)out->data[0]);

    av_frame_free(&in);
    gs->frame_idx++;
    av_log(ctx, AV_LOG_VERBOSE, "filter_frame end\n");
    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx) {
  GenericShaderContext *gs = ctx->priv;
  glDeleteTextures(1, &gs->frame_tex);
  glDeleteProgram(gs->program);
  glDeleteBuffers(1, &gs->pos_buf);
  glfwDestroyWindow(gs->window);
  av_expr_free(gs->power_pexpr); gs->power_pexpr = NULL;
}

static int query_formats(AVFilterContext *ctx) {
  static const enum AVPixelFormat formats[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};
  return ff_set_common_formats(ctx, ff_make_format_list(formats));
}

#define OFFSET(x) offsetof(GenericShaderContext, x)
static const AVOption genericshader_options[] = {
    {"vs_textfile",    "set a text file for vertex shader",        OFFSET(vs_textfile),           AV_OPT_TYPE_STRING, {.str=NULL},  CHAR_MIN, CHAR_MAX, FLAGS},
    {"fs_textfile",    "set a text file for fragment shader",      OFFSET(fs_textfile),           AV_OPT_TYPE_STRING, {.str=NULL},  CHAR_MIN, CHAR_MAX, FLAGS},
    { "power", "set the power expression", OFFSET(power_expr), AV_OPT_TYPE_STRING, {.str = "0"}, CHAR_MIN, CHAR_MAX, FLAGS },
    { "shader_style", "set the shader", OFFSET(shader_style), AV_OPT_TYPE_STRING, {.str = "0"}, CHAR_MIN, CHAR_MAX, FLAGS },
    { "eval", "specify when to evaluate expressions", OFFSET(eval_mode), AV_OPT_TYPE_INT, {.i64 = EVAL_MODE_FRAME}, 0, EVAL_MODE_NB-1, FLAGS, "eval" },
             { "init",  "eval expressions once during initialization", 0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_INIT},  .flags = FLAGS, .unit = "eval" },
             { "frame", "eval expressions per-frame",                  0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_FRAME}, .flags = FLAGS, .unit = "eval" },
    { "is_color", "relevant to vintage, specify color mode", OFFSET(is_color), AV_OPT_TYPE_INT, {.i64 = IS_COLOR_MODE_TRUE}, 0, IS_COLOR_MODE_NB-1, FLAGS, "eval" },
             { "true",  "color mode",    0, AV_OPT_TYPE_CONST, {.i64=IS_COLOR_MODE_TRUE},  .flags = FLAGS, .unit = "eval" },
             { "false", "no color mode", 0, AV_OPT_TYPE_CONST, {.i64=IS_COLOR_MODE_FALSE}, .flags = FLAGS, .unit = "eval" },
    { "drop_size",  "matrix drop size", OFFSET(dropSize), AV_OPT_TYPE_FLOAT, {.dbl=5.0}, 0, 1, FLAGS },
    {NULL}
};

AVFILTER_DEFINE_CLASS(genericshader);

static const AVFilterPad genericshader_inputs[] = {
  {.name = "default",
   .type = AVMEDIA_TYPE_VIDEO,
   .config_props = config_props,
   .filter_frame = filter_frame},
  {NULL}};

static const AVFilterPad genericshader_outputs[] = {
  {.name = "default", .type = AVMEDIA_TYPE_VIDEO}, {NULL}};

AVFilter ff_vf_genericshader = {
  .name          = "genericshader",
  .description   = NULL_IF_CONFIG_SMALL("Generic OpenGL shader filter"),
  .priv_size     = sizeof(GenericShaderContext),
  .priv_class    = &genericshader_class,
  .init          = init,
  .uninit        = uninit,
  .query_formats = query_formats,
  .process_command = process_command,
  .inputs        = genericshader_inputs,
  .outputs       = genericshader_outputs,
  .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC};