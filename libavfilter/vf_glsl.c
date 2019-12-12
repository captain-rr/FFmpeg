#include <string.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
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
#include "framesync.h"

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

typedef struct GLSLContext {
    const AVClass *class;

	// internal state
	GLuint        program;
	//GLuint        active_program;
	//GLuint        *programs;
    GLuint        frame_tex;
    GLFWwindow    *window;
    GLuint        pos_buf;
    int           frame_idx;

	// input options
	int eval_mode;		///< EvalMode
	int shader;			///< ShaderTypes
	int is_color;		// for vintage filter
    float dropSize;		// for matrix filter
	char *power_expr;	// power string expression
	char *vs_textfile;
	char *fs_textfile;

	// file contents
    uint8_t *vs_text;
    uint8_t *fs_text;

    double var_values[VAR_VARS_NB];
	AVExpr *power_pexpr; // power expression struct
	float power; // power value

	// transition context
		FFFrameSync fs;

		// input options
		double duration;
		double offset;
		char *transition_source;

		// timestamp of the first frame in the output, in the timebase units
		int64_t first_pts;

		// uniforms
		GLuint        uFrom;
		GLuint        uTo;

		GLchar *f_shader_source;
	// transition context
} GLSLContext;

#define OFFSET(x) offsetof(GLSLContext, x)

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

enum ShaderTypes {
	SHADER_TYPE_PASSTHROUGH,
	SHADER_TYPE_MATRIX,
	SHADER_TYPE_SHOCKWAVE,
	SHADER_TYPE_VINTAGE,
	SHADER_TYPE_TRANSITION,
	SHADER_TYPE_NB,
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

static const GLchar *v_overlay_shader_source = 
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

static const GLchar *f_transition_shader_template =
"varying vec2 texCoord;\n"
"uniform sampler2D from;\n"
"uniform sampler2D to;\n"
"uniform float progress;\n"
"\n"
"vec4 getFromColor(vec2 uv) {\n"
"  return texture2D(from, uv);\n"
"}\n"

"vec4 getToColor(vec2 uv) {\n"
"  return texture2D(to, uv);\n"
"}\n"
"\n"
"\n%s\n"
"void main() {\n"
"  gl_FragColor = transition(texCoord);\n"
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
"uniform sampler2D tex;\n"
"uniform float power;\n"
"uniform float time;\n"
"uniform bool isColor;\n"

"void main() {\n"
"    vec2 uv = texCoord;\n"
"    vec4 color = texture2D(tex, uv);\n"
"    vec4 originalColor = texture2D(tex, uv);\n"
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

// default to a basic fade effect
static const uint8_t *f_default_transition_source =
"vec4 transition (vec2 uv) {\n"
"  return mix(\n"
"    getFromColor(uv),\n"
"    getToColor(uv),\n"
"    power\n"
"  );\n"
"}\n";

#define PIXEL_FORMAT GL_RGB
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
#define MAIN    0
#define FROM (0)
#define TO   (1)

static inline float normalize_power(double d)
{
    if (isnan(d))
        return FLT_MAX;
    return (float)d;
}

static void eval_expr(AVFilterContext *ctx)
{
	GLSLContext *s = ctx->priv;
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
	GLSLContext *s = ctx->priv;

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
           "load_textfile '%s' size:%d 2\n",
           textfile, textbuf_size);

    if (textbuf_size > SIZE_MAX - 1 ||
        !(tmp = av_realloc(*text, textbuf_size + 1))) {
		av_log(ctx, AV_LOG_ERROR,
			"The text file '%s' created buffer size issue\n",
			textfile);
        av_file_unmap(textbuf, textbuf_size);
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
    (*text)[textbuf_size] = 0;
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

static void setup_vbo(GLSLContext *c, AVFilterContext *log_ctx) {
  GLint loc;
  glGenBuffers(1, &c->pos_buf);
  glBindBuffer(GL_ARRAY_BUFFER, c->pos_buf);
  glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);

  loc = glGetAttribLocation(c->program, "position");
  glEnableVertexAttribArray(loc);
  glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

static void setup_tex(AVFilterLink *inlink) {
    AVFilterContext     *ctx = inlink->dst;
	GLSLContext *c = ctx->priv;

	if (c->shader == SHADER_TYPE_TRANSITION) {
		{ // from
			glGenTextures(1, &c->uFrom);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, c->uFrom);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, NULL);

			glUniform1i(glGetUniformLocation(c->program, "from"), 0);
		}

		{ // to
			glGenTextures(1, &c->uTo);
			glActiveTexture(GL_TEXTURE0 + 1);
			glBindTexture(GL_TEXTURE_2D, c->uTo);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, NULL);

			glUniform1i(glGetUniformLocation(c->program, "to"), 1);
		}
	}
	else {
		glGenTextures(1, &c->frame_tex);
		glActiveTexture(GL_TEXTURE0);

		glBindTexture(GL_TEXTURE_2D, c->frame_tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, NULL);

		glUniform1i(glGetUniformLocation(c->program, "tex"), 0);
	}
}

static void setup_uniforms(AVFilterLink *fromLink)
{
	AVFilterContext     *ctx = fromLink->dst;
	GLSLContext *c = ctx->priv;

	glUniform1f(glGetUniformLocation(c->program, "power"), 0.0f);

	if (c->shader == SHADER_TYPE_MATRIX) {
		glUniform1f(glGetUniformLocation(c->program, "time"), 0.0f);
		glUniform1f(glGetUniformLocation(c->program, "dropSize"), 0.0f);
	}
	else if (c->shader == SHADER_TYPE_SHOCKWAVE) {
		glUniform1f(glGetUniformLocation(c->program, "time"), 0.0f);
	}
	else if (c->shader == SHADER_TYPE_VINTAGE) {
		glUniform1f(glGetUniformLocation(c->program, "time"), 0.0f);
		glUniform1i(glGetUniformLocation(c->program, "isColor"), 0);
	}
	//else if (c->shader == SHADER_TYPE_TRANSITION) {

	//}
}

static int build_program(AVFilterContext *ctx) {
    GLint status;
    GLuint v_shader, f_shader;
    GLchar* shader;
	GLSLContext *c = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "build_program %s\n", c->shader);

	if (c->shader == SHADER_TYPE_TRANSITION) {
		if (!(v_shader = build_shader(ctx, v_shader_source, GL_VERTEX_SHADER))) {
			av_log(ctx, AV_LOG_ERROR, "invalid vertex shader\n");
			return -1;
		}

		if (!(f_shader = build_shader(ctx, c->f_shader_source, GL_FRAGMENT_SHADER))) {
			av_log(ctx, AV_LOG_ERROR, "invalid fragment shader\n");
			return -1;
		}
	}
	else {
		if (c->vs_text) {
			av_log(ctx, AV_LOG_VERBOSE, "build_program vs_from_text ||%s||\n", c->vs_text);
			v_shader = build_shader(ctx, (GLchar*)c->vs_text, GL_VERTEX_SHADER);
		}
		else {
			v_shader = build_shader(ctx, v_shader_source, GL_VERTEX_SHADER);
		}

		if (c->fs_text) {
			av_log(ctx, AV_LOG_VERBOSE, "build_program_2_s %d ||%s||\n", v_shader, c->fs_text);
			f_shader = build_shader(ctx, (GLchar*)c->fs_text, GL_FRAGMENT_SHADER);
		}
		else if (c->shader == SHADER_TYPE_MATRIX) {
			f_shader = build_shader(ctx, f_matrix_shader_source, GL_FRAGMENT_SHADER);
		}
		else if (c->shader == SHADER_TYPE_SHOCKWAVE) {
			f_shader = build_shader(ctx, f_shockwave_shader_source, GL_FRAGMENT_SHADER);
		}
		else if (c->shader == SHADER_TYPE_VINTAGE) {
			f_shader = build_shader(ctx, f_vintage_shader_source, GL_FRAGMENT_SHADER);
		}
		else {
			f_shader = build_shader(ctx, f_shader_source, GL_FRAGMENT_SHADER);
		}

		if (!(v_shader && f_shader)) {
			av_log(ctx, AV_LOG_VERBOSE, "build_program shader build fail %d %d\n", v_shader, f_shader);
			return -1;
		}
	}

	c->program = glCreateProgram();
	glAttachShader(c->program, v_shader);
	glAttachShader(c->program, f_shader);
	glLinkProgram(c->program);

	glGetProgramiv(c->program, GL_LINK_STATUS, &status);
	return status == GL_TRUE ? 0 : -1;
}

static av_cold int init(AVFilterContext *ctx) {
    int err;
    int status;
	GLSLContext *c = ctx->priv;
    av_log(ctx, AV_LOG_VERBOSE, "init\n");
    if (c->vs_textfile) {
        av_log(ctx, AV_LOG_VERBOSE, "attempt load text file for vertex shader '%s'\n", c->vs_textfile);
        if ((err = load_textfile(ctx, c->vs_textfile, &c->vs_text)) < 0)
            return err;
    }
    if (c->fs_textfile) {
        av_log(ctx, AV_LOG_VERBOSE, "attempt load text file for fragment shader '%s'\n", c->fs_textfile);
        if ((err = load_textfile(ctx, c->fs_textfile, &c->fs_text)) < 0)
            return err;
    }
    if (!c->shader) {
        av_log(ctx, AV_LOG_ERROR, "Empty output shader style string.\n");
        return AVERROR(EINVAL);
    }

    status = glfwInit();
    av_log(ctx, AV_LOG_VERBOSE, "GLFW init status:%d\n", status);
    return status? 0 : -1;
}

static AVFrame *apply_transition(FFFrameSync *fs,
	AVFilterContext *ctx,
	AVFrame *fromFrame,
	const AVFrame *toFrame)
{
	GLSLContext *c = ctx->priv;
	AVFilterLink *fromLink = ctx->inputs[FROM];
	AVFilterLink *toLink = ctx->inputs[TO];
	AVFilterLink *outLink = ctx->outputs[0];
	AVFrame *outFrame;

	outFrame = ff_get_video_buffer(outLink, outLink->w, outLink->h);
	if (!outFrame) {
		return NULL;
	}

	av_frame_copy_props(outFrame, fromFrame);

	glfwMakeContextCurrent(c->window);

	glUseProgram(c->program);

	const float ts = ((fs->pts - c->first_pts) / (float)fs->time_base.den) - c->offset;
	const float power = FFMAX(0.0f, FFMIN(1.0f, ts / c->duration));
	// av_log(ctx, AV_LOG_ERROR, "transition '%s' %llu %f %f\n", c->transition_source, fs->pts - c->first_pts, ts, power);
	glUniform1f(glGetUniformLocation(c->program, "power"), power);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, c->uFrom);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, fromFrame->linesize[0] / 3);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fromLink->w, fromLink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, fromFrame->data[0]);

	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, c->uTo);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, toFrame->linesize[0] / 3);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, toLink->w, toLink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, toFrame->data[0]);

	glDrawArrays(GL_TRIANGLES, 0, 6);
	glPixelStorei(GL_PACK_ROW_LENGTH, outFrame->linesize[0] / 3);
	glReadPixels(0, 0, outLink->w, outLink->h, PIXEL_FORMAT, GL_UNSIGNED_BYTE, (GLvoid *)outFrame->data[0]);

	glPixelStorei(GL_PACK_ROW_LENGTH, 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

	av_frame_free(&fromFrame);

	return outFrame;
}

static int blend_frame(FFFrameSync *fs)
{
	AVFilterContext *ctx = fs->parent;
	GLSLContext *c = ctx->priv;

	AVFrame *fromFrame, *toFrame, *outFrame;
	int ret;

	ret = ff_framesync_dualinput_get(fs, &fromFrame, &toFrame);
	if (ret < 0) {
		return ret;
	}

	if (c->first_pts == AV_NOPTS_VALUE &&
		fromFrame &&
		fromFrame->pts != AV_NOPTS_VALUE) {
		c->first_pts = fromFrame->pts;
	}

	if (!toFrame) {
		return ff_filter_frame(ctx->outputs[0], fromFrame);
	}

	outFrame = apply_transition(fs, ctx, fromFrame, toFrame);
	if (!outFrame) {
		return AVERROR(ENOMEM);
	}

	return ff_filter_frame(ctx->outputs[0], outFrame);
}

static av_cold int init_transition(AVFilterContext *ctx)
{
	int err, status;
	GLSLContext *c;
	uint8_t *transition_function;

	c = ctx->priv;
	av_log(ctx, AV_LOG_VERBOSE, "init_transition %d\n", c->shader);
	c->fs.on_event = blend_frame;
	c->first_pts = AV_NOPTS_VALUE;
	c->shader = SHADER_TYPE_TRANSITION;

	if (c->transition_source) {
		av_log(ctx, AV_LOG_VERBOSE, "attempt load text file for transition function '%s'\n", c->transition_source);
		if ((err = load_textfile(ctx, c->transition_source, &transition_function)) < 0)
			return err;
	}
	else 
	{
		transition_function = f_default_transition_source;
	}

	int len = strlen(f_transition_shader_template) + strlen((char *)transition_function);
	c->f_shader_source = av_calloc(len, sizeof(*c->f_shader_source));
	if (!c->f_shader_source) {
		av_log(ctx, AV_LOG_ERROR, "failed alloaction f_shader_source\n");
		return AVERROR(ENOMEM);
	}

	snprintf(c->f_shader_source, len * sizeof(*c->f_shader_source), f_transition_shader_template, transition_function);
	av_log(ctx, AV_LOG_DEBUG, "\n%s\n", c->f_shader_source);

	if (c->transition_source) {
		free(transition_function);
		transition_function = NULL;
	}

	status = glfwInit();
	av_log(ctx, AV_LOG_VERBOSE, "GLFW init status:%d\n", status);
	return status ? 0 : -1;
}

static av_cold void uninit_transition(AVFilterContext *ctx) {
	GLSLContext *c = ctx->priv;
	ff_framesync_uninit(&c->fs);

	if (c->window) {
		glDeleteTextures(1, &c->uFrom);
		glDeleteTextures(1, &c->uTo);
		glDeleteBuffers(1, &c->pos_buf);
		glDeleteProgram(c->program);
		glfwDestroyWindow(c->window);
	}

	if (c->f_shader_source) {
		av_freep(&c->f_shader_source);
	}
}

static int config_input_props(AVFilterLink *inlink) {
	int ret;
	AVFilterContext     *ctx = inlink->dst;
	GLSLContext *c = ctx->priv;

	//glfw
	glfwWindowHint(GLFW_VISIBLE, 0);
	c->window = glfwCreateWindow(inlink->w, inlink->h, "", NULL, NULL);
	if (!c->window) {
		av_log(ctx, AV_LOG_ERROR, "setup_gl ERROR");
		return -1;
	}
	glfwMakeContextCurrent(c->window);

#ifndef __APPLE__
	glewExperimental = GL_TRUE;
	glewInit();
#endif

	glViewport(0, 0, inlink->w, inlink->h);

	if ((ret = build_program(ctx)) < 0) {
		return ret;
	}
	c->var_values[VAR_MAIN_W] = c->var_values[VAR_MW] = ctx->inputs[MAIN]->w;
	c->var_values[VAR_MAIN_H] = c->var_values[VAR_MH] = ctx->inputs[MAIN]->h;
	c->var_values[VAR_POWER] = NAN;
	c->var_values[VAR_T] = NAN;
	c->var_values[VAR_POS] = NAN;

	if ((ret = set_expr(&c->power_pexpr, c->power_expr, "power", ctx)) < 0)
		return ret;

	if (c->eval_mode == EVAL_MODE_INIT) {
		eval_expr(ctx);
		av_log(ctx, AV_LOG_INFO, "pow:%f powi:%f\n",
			c->var_values[VAR_POWER], c->power);
	}

	av_log(ctx, AV_LOG_VERBOSE,
		"main w:%d h:%d\n",
		ctx->inputs[MAIN]->w, ctx->inputs[MAIN]->h,
		av_get_pix_fmt_name(ctx->inputs[MAIN]->format));

	glUseProgram(c->program);
	setup_vbo(c, ctx);
	setup_uniforms(inlink);
	setup_tex(inlink);

	return 0;
}

static int config_transition_output(AVFilterLink *outLink)
{
	AVFilterContext *ctx = outLink->src;
	GLSLContext *c = ctx->priv;
	AVFilterLink *fromLink = ctx->inputs[FROM];
	AVFilterLink *toLink = ctx->inputs[TO];
	int ret;

	if (fromLink->format != toLink->format) {
		av_log(ctx, AV_LOG_ERROR, "inputs must be of same pixel format\n");
		return AVERROR(EINVAL);
	}

	if (fromLink->w != toLink->w || fromLink->h != toLink->h) {
		av_log(ctx, AV_LOG_ERROR, "First input link %s parameters "
			"(size %dx%d) do not match the corresponding "
			"second input link %s parameters (size %dx%d)\n",
			ctx->input_pads[FROM].name, fromLink->w, fromLink->h,
			ctx->input_pads[TO].name, toLink->w, toLink->h);
		return AVERROR(EINVAL);
	}

	outLink->w = fromLink->w;
	outLink->h = fromLink->h;
	// outLink->time_base = fromLink->time_base;
	outLink->frame_rate = fromLink->frame_rate;

	if ((ret = ff_framesync_init_dualinput(&c->fs, ctx)) < 0) {
		return ret;
	}

	return ff_framesync_configure(&c->fs);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in) {
    AVFilterContext *ctx     = inlink->dst;
    AVFilterLink    *outlink = ctx->outputs[0];
    GLSLContext *c = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "filter_frame\n");

    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    if (c->eval_mode == EVAL_MODE_FRAME) {
      int64_t pos = in->pkt_pos;

      c->var_values[VAR_N] = inlink->frame_count_out;
      c->var_values[VAR_T] = in->pts == AV_NOPTS_VALUE ? NAN : in->pts * av_q2d(inlink->time_base);
      c->var_values[VAR_POS] = pos == -1 ? NAN : pos;

      c->var_values[VAR_MAIN_W] = c->var_values[VAR_MW] = in->width;
      c->var_values[VAR_MAIN_H] = c->var_values[VAR_MH] = in->height;

      eval_expr(ctx);
      av_log(ctx, AV_LOG_VERBOSE, "filter_frame pow:%f powi:%f time:%f\n",
             c->var_values[VAR_POWER], c->power, c->var_values[VAR_T]);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, in->data[0]);
    if (c->shader == SHADER_TYPE_MATRIX){
        glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &c->power);
        GLfloat time = (GLfloat)(c->var_values[VAR_T] == NAN? c->frame_idx * 330: c->var_values[VAR_T] * 1000);
        GLfloat dropSize = (GLfloat)c->dropSize;
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame matrix time:%f dropSize:%f\n", time, dropSize);
        glUniform1fv(glGetUniformLocation(c->program, "time"), 1, &time);
        glUniform1fv(glGetUniformLocation(c->program, "dropSize"), 1, &dropSize);
    } else if (c->shader == SHADER_TYPE_SHOCKWAVE){
        glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &c->power);
        GLfloat time = (GLfloat)(c->var_values[VAR_T] == NAN? c->frame_idx * 1.6667: c->var_values[VAR_T] * 5);
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame shockwave time:%f\n", time);
        glUniform1fv(glGetUniformLocation(c->program, "time"), 1, &time);
    } else if (c->shader == SHADER_TYPE_VINTAGE){
        glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &c->power);
        GLfloat time = (GLfloat)(c->var_values[VAR_T] == NAN? c->frame_idx * 330: c->var_values[VAR_T] * 1000);
        GLint isColor = c->is_color == IS_COLOR_MODE_TRUE? 1: 0;
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame vintage isColor:%d time:%f\n", isColor, time);
        glUniform1fv(glGetUniformLocation(c->program, "time"), 1, &time);
        glUniform1iv(glGetUniformLocation(c->program, "isColor"), 1, &isColor);
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glReadPixels(0, 0, outlink->w, outlink->h, PIXEL_FORMAT, GL_UNSIGNED_BYTE, (GLvoid *)out->data[0]);

    av_frame_free(&in);
    c->frame_idx++;
    av_log(ctx, AV_LOG_VERBOSE, "filter_frame end\n");
    return ff_filter_frame(outlink, out);
}


static av_cold void uninit(AVFilterContext *ctx) {
    GLSLContext *c = ctx->priv;
    av_log(ctx, AV_LOG_VERBOSE, "uninit\n");

    glDeleteTextures(1, &c->frame_tex);
    av_log(ctx, AV_LOG_VERBOSE, "uninit1\n");
    if (c->program){
        glDeleteProgram(c->program);
        av_log(ctx, AV_LOG_VERBOSE, "uninit2\n");
        glDeleteBuffers(1, &c->pos_buf);
        av_log(ctx, AV_LOG_VERBOSE, "uninit3\n");
    }
    if (c->window){
        glfwDestroyWindow(c->window);
        av_log(ctx, AV_LOG_VERBOSE, "uninit4\n");
    }
    if (c->power_pexpr){
        av_expr_free(c->power_pexpr);
        av_log(ctx, AV_LOG_VERBOSE, "uninit5\n");
        c->power_pexpr = NULL;
    }
    av_log(ctx, AV_LOG_VERBOSE, "uninit6\n");
}

//necessary for transition only because of the f-sync
static int activate_transition(AVFilterContext *ctx)
{
	GLSLContext *c = ctx->priv;
	return ff_framesync_activate(&c->fs);
}

static int query_formats(AVFilterContext *ctx)
{
	static const enum AVPixelFormat formats[] = {
	  AV_PIX_FMT_RGB24,
	  AV_PIX_FMT_NONE
	};

	return ff_set_common_formats(ctx, ff_make_format_list(formats));
}

static const AVOption glsl_options[] = {
	{ "shader", "set the shader", OFFSET(shader), AV_OPT_TYPE_INT, {.i64 = SHADER_TYPE_PASSTHROUGH}, 0, SHADER_TYPE_NB-1, FLAGS, "shader" },
			 { "matrix",  "set matrix like effect",    0, AV_OPT_TYPE_CONST, {.i64 = SHADER_TYPE_MATRIX},.flags = FLAGS,.unit = "shader" },
			 { "shockwave", "set shockwave like effect", 0, AV_OPT_TYPE_CONST, {.i64 = SHADER_TYPE_SHOCKWAVE},.flags = FLAGS,.unit = "shader" },
			 { "vintage", "set vintage like effect", 0, AV_OPT_TYPE_CONST, {.i64 = SHADER_TYPE_VINTAGE},.flags = FLAGS,.unit = "shader" },
	{ "vs_textfile",    "set a text file for vertex shader",        OFFSET(vs_textfile),           AV_OPT_TYPE_STRING, {.str=NULL},  CHAR_MIN, CHAR_MAX, FLAGS},
    { "fs_textfile",    "set a text file for fragment shader",      OFFSET(fs_textfile),           AV_OPT_TYPE_STRING, {.str=NULL},  CHAR_MIN, CHAR_MAX, FLAGS},
    { "power", "set the power expression", OFFSET(power_expr), AV_OPT_TYPE_STRING, {.str = "0"}, CHAR_MIN, CHAR_MAX, FLAGS },
    { "eval", "specify when to evaluate expressions", OFFSET(eval_mode), AV_OPT_TYPE_INT, {.i64 = EVAL_MODE_FRAME}, 0, EVAL_MODE_NB-1, FLAGS, "eval" },
             { "init",  "eval expressions once during initialization", 0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_INIT},  .flags = FLAGS, .unit = "eval" },
             { "frame", "eval expressions per-frame",                  0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_FRAME}, .flags = FLAGS, .unit = "eval" },
    { "is_color", "relevant to vintage, specify color mode", OFFSET(is_color), AV_OPT_TYPE_INT, {.i64 = IS_COLOR_MODE_TRUE}, 0, IS_COLOR_MODE_NB-1, FLAGS, "is_color" },
             { "true",  "color mode",    0, AV_OPT_TYPE_CONST, {.i64=IS_COLOR_MODE_TRUE},  .flags = FLAGS, .unit = "is_color" },
             { "false", "no color mode", 0, AV_OPT_TYPE_CONST, {.i64=IS_COLOR_MODE_FALSE}, .flags = FLAGS, .unit = "is_color" },
    { "drop_size",  "matrix drop size", OFFSET(dropSize), AV_OPT_TYPE_FLOAT, {.dbl=5.0}, 0, 100, FLAGS },
    {NULL}
};

static const AVFilterPad glsl_inputs[] = {
	{.name = "default",
	.type = AVMEDIA_TYPE_VIDEO,
	.config_props = config_input_props,
	.filter_frame = filter_frame},
	{NULL}
};

static const AVFilterPad glsl_outputs[] = {
	{.name = "default",.type = AVMEDIA_TYPE_VIDEO},
	{NULL}
};

AVFILTER_DEFINE_CLASS(glsl);

AVFilter ff_vf_glsl = {
  .name          = "glsl",
  .description   = NULL_IF_CONFIG_SMALL("Generic OpenGL shader filter"),
  .priv_size     = sizeof(GLSLContext),
  .priv_class    = &glsl_class,
  .init          = init,
  .uninit        = uninit,
  .query_formats = query_formats,
  .process_command = process_command,
  .inputs        = glsl_inputs,
  .outputs       = glsl_outputs,
  .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC};


static const AVOption gltransition_options[] = {
	{ "duration", "transition duration in seconds", OFFSET(duration), AV_OPT_TYPE_DOUBLE, {.dbl = 1.0}, 0, DBL_MAX, FLAGS },
	{ "offset", "delay before startingtransition in seconds", OFFSET(offset), AV_OPT_TYPE_DOUBLE, {.dbl = 0.0}, 0, DBL_MAX, FLAGS },
	{ "transition", "path to the gl-transition source file (defaults to basic fade)", OFFSET(transition_source), AV_OPT_TYPE_STRING, {.str = NULL}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "shader", "should always be left at default value", OFFSET(shader), AV_OPT_TYPE_INT, {.i64 = SHADER_TYPE_TRANSITION}, SHADER_TYPE_TRANSITION, SHADER_TYPE_TRANSITION, FLAGS },
	{NULL}
};

static const AVFilterPad gltransition_inputs[] = {
  {
	.name = "from",
	.type = AVMEDIA_TYPE_VIDEO,
	.config_props = config_input_props,
  },
  {
	.name = "to",
	.type = AVMEDIA_TYPE_VIDEO,
  },
  {NULL}
};

static const AVFilterPad gltransition_outputs[] = {
  {
	.name = "default",
	.type = AVMEDIA_TYPE_VIDEO,
	.config_props = config_transition_output,
  },
  {NULL}
};

FRAMESYNC_DEFINE_CLASS(gltransition, GLSLContext, fs);

AVFilter ff_vf_gltransition = {
  .name = "gltransition",
  .description = NULL_IF_CONFIG_SMALL("OpenGL blend transitions"),
  .priv_size = sizeof(GLSLContext),
  .preinit = gltransition_framesync_preinit,
  .init = init_transition,
  .uninit = uninit_transition,
  .query_formats = query_formats,
  .activate = activate_transition,
  .inputs = gltransition_inputs,
  .outputs = gltransition_outputs,
  .priv_class = &gltransition_class,
  .flags = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC
};