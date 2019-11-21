#include <string.h>
#include <float.h>
#include "internal.h"

#include "avfilter.h"
#include "formats.h"
#include "libavutil/common.h"
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

    float power;

    int eval_mode;              ///< EvalMode

    double var_values[VAR_VARS_NB];
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
  "  gl_Position = vec4(position* 0.5 + 0.5, 0, 1);\n"
  "  texCoord = position;\n"
  "}\n";

static const GLchar *f_shader_source =
  "varying vec2 texCoord;\n"
  "uniform sampler2D tex;\n"
  "void main() {\n"
  "  gl_FragColor = texture2D(tex, texCoord);\n"
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

"void main() {\n"
"    vec4 originalColor = texture2D(tex, texCoord);\n"
"    float globalTime = time * RAIN_SPEED + 34284.0;\n"
"    vec2 position = - texCoord;\n"
"    vec2 res = vec2(700,300);\n"
"    position.x /= res.x / res.y;\n"
"    float scaledown = DROP_SIZE;\n"
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

"    scaledown = DROP_SIZE;\n"
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
  }
  av_log(log_ctx, AV_LOG_VERBOSE, "tex_setup end\n");
}

static int build_program(AVFilterContext *ctx) {
    GLint status;
    GLuint v_shader, f_shader;
    GLchar* shader;
    GenericShaderContext *gs = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "build_program %s\n", gs->shader_style);

    v_shader = build_shader(ctx, v_shader_source, GL_VERTEX_SHADER);

    if (!strcmp(gs->shader_style, "matrix")){
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_1 %d\n", v_shader);
        f_shader = build_shader(ctx, f_matrix_shader_source, GL_FRAGMENT_SHADER);
    } else {
        av_log(ctx, AV_LOG_VERBOSE, "build_program_2_2 %d\n", v_shader);
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
    GenericShaderContext *gs = ctx->priv;
    av_log(ctx, AV_LOG_VERBOSE, "init\n");
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
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame matrix time:%f\n", time);
        glUniform1fv(glGetUniformLocation(gs->program, "time"), 1, &time);
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
    { "power", "set the power expression", OFFSET(power_expr), AV_OPT_TYPE_STRING, {.str = "0"}, CHAR_MIN, CHAR_MAX, FLAGS },
    { "shader_style", "set the shader", OFFSET(shader_style), AV_OPT_TYPE_STRING, {.str = "0"}, CHAR_MIN, CHAR_MAX, FLAGS },
    { "eval", "specify when to evaluate expressions", OFFSET(eval_mode), AV_OPT_TYPE_INT, {.i64 = EVAL_MODE_FRAME}, 0, EVAL_MODE_NB-1, FLAGS, "eval" },
             { "init",  "eval expressions once during initialization", 0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_INIT},  .flags = FLAGS, .unit = "eval" },
             { "frame", "eval expressions per-frame",                  0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_FRAME}, .flags = FLAGS, .unit = "eval" },
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