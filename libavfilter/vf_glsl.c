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
	
	char *r_expr;	// r string expression
	char *g_expr;	// g string expression
	char *b_expr;	// b string expression
	
	char *brightness_expr;	// brightness string expression
	char *contrast_expr;	// contrast string expression
	char *saturation_expr;	// saturation string expression
	
	char *vs_textfile;
	char *fs_textfile;

	// file contents
    char *vs_text;
	char *fs_text;

    double var_values[VAR_VARS_NB];
	AVExpr *power_pexpr; // power expression struct
	float power; // power value

	AVExpr *r_pexpr; // red expression struct
	float adjust_r; // adjust red value
	AVExpr *g_pexpr; // green expression struct
	float adjust_g; // adjust green value
	AVExpr *b_pexpr; // blue expression struct
	float adjust_b; // adjust blue value

	AVExpr *brightness_pexpr; // brightness expression struct
	float brightness; // adjust brightness value
	AVExpr *contrast_pexpr; // contrast expression struct
	float contrast; // adjust contrast value
	AVExpr *saturation_pexpr; // saturation expression struct
	float saturation; // adjust saturation value

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
    SHADER_TYPE_ADJUST,
    SHADER_TYPE_OLD_FILM,
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
"uniform float power;\n"
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

static const GLchar *f_old_film_shader_source =
    "varying vec2 texCoord;\n"
    "uniform sampler2D tex;\n"

    "const vec2 dimensions = vec2(1280, 720);\n"

    "const float noise = 0.3;\n"
    "const float noiseSize = 1.0;\n"
    "const float scratch = 0.5;\n"
    "const float scratchDensity = 0.3;\n"
    "const float scratchWidth = 1.0;\n"
    "const float vignettingAlpha = 1.0;\n"
    "const float vignettingBlur = 0.3;\n"
	"const float sepia = 0.5;\n"
	"const float vignetting = 0.4;\n"

    "uniform float power;\n"
    
    "const float SQRT_2 = 1.414213;\n"
    "const vec3 SEPIA_RGB = vec3(112.0 / 255.0, 66.0 / 255.0, 20.0 / 255.0);\n"

    "float rand(vec2 co) {\n"
    "    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);\n"
    "}\n"

    "vec3 Overlay(vec3 src, vec3 dst)\n"
    "{\n"
    "    // if (dst <= 0.5) then: 2 * src * dst\n"
    "    // if (dst > 0.5) then: 1 - 2 * (1 - dst) * (1 - src)\n"
    "    return vec3((dst.x <= 0.5) ? (2.0 * src.x * dst.x) : (1.0 - 2.0 * (1.0 - dst.x) * (1.0 - src.x)),\n"
    "                (dst.y <= 0.5) ? (2.0 * src.y * dst.y) : (1.0 - 2.0 * (1.0 - dst.y) * (1.0 - src.y)),\n"
    "                (dst.z <= 0.5) ? (2.0 * src.z * dst.z) : (1.0 - 2.0 * (1.0 - dst.z) * (1.0 - src.z)));\n"
    "}\n"

    "void main()\n"
    "{\n"
    "    gl_FragColor = texture2D(tex, texCoord);\n"
    "    vec3 color = gl_FragColor.rgb;\n"
    "    float seed = power;\n"
    "    vec2 coord = texCoord;\n"

    "    if (sepia > 0.0)\n"
    "    {\n"
    "        float gray = (color.x + color.y + color.z) / 3.0;\n"
    "        vec3 grayscale = vec3(gray);\n"
    "        color = Overlay(SEPIA_RGB, grayscale);\n"
    "        color = grayscale + sepia * (color - grayscale);\n"
    "    }\n"

    "    if (vignetting > 0.0)\n"
    "    {\n"
    "        float outter = SQRT_2 - vignetting * SQRT_2;\n"
    "        vec2 dir = vec2(vec2(0.5, 0.5) - coord);\n"
    "        dir.y *= dimensions.y / dimensions.x;\n"
    "        float darker = clamp((outter - length(dir) * SQRT_2) / ( 0.00001 + vignettingBlur * SQRT_2), 0.0, 1.0);\n"
    "        color.rgb *= darker + (1.0 - darker) * (1.0 - vignettingAlpha);\n"
    "    }\n"

    "    if (scratchDensity > seed && scratch != 0.0)\n"
    "    {\n"
    "        float phase = seed * 256.0;\n"
    "        float s = mod(floor(phase), 2.0);\n"
    "        float dist = 1.0 / scratchDensity;\n"
    "        float d = distance(coord, vec2(seed * dist, abs(s - seed * dist)));\n"
    "        if (d < seed * 0.6 + 0.4)\n"
    "        {\n"
    "            highp float period = scratchDensity * 10.0;\n"

    "            float xx = coord.x * period + phase;\n"
    "            float aa = abs(mod(xx, 0.5) * 4.0);\n"
    "            float bb = mod(floor(xx / 0.5), 2.0);\n"
    "            float yy = (1.0 - bb) * aa + bb * (2.0 - aa);\n"

    "            float kk = 2.0 * period;\n"
    "            float dw = scratchWidth / dimensions.x * (0.75 + seed);\n"
    "            float dh = dw * kk;\n"
    "            float tine = (yy - (2.0 - dh));\n"

    "            if (tine > 0.0) {\n"
    "                float _sign = sign(scratch);\n"
    "                tine = s * tine / period + scratch + 0.1;\n"
    "                tine = clamp(tine + 1.0, 0.5 + _sign * 0.5, 1.5 + _sign * 0.5);\n"
    "                color.rgb *= tine;\n"
    "            }\n"
    "        }\n"
    "    }\n"

    "    if (noise > 0.0 && noiseSize > 0.0)\n"
    "    {\n"
    "        vec2 pixelCoord = texCoord.xy * dimensions.xy;\n"
    "        pixelCoord.x = floor(pixelCoord.x / noiseSize);\n"
    "        pixelCoord.y = floor(pixelCoord.y / noiseSize);\n"
    "        // vec2 d = pixelCoord * noiseSize * vec2(1024.0 + seed * 512.0, 1024.0 - seed * 512.0);\n"
    "        // float _noise = snoise(d) * 0.5;\n"
    "        float _noise = rand(pixelCoord * noiseSize * seed) - 0.5;\n"
    "        color += _noise * noise;\n"
    "    }\n"

    "    gl_FragColor.rgb = color;\n"
    "}";

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

static const GLchar *f_adjust_shader_source =
"varying vec2 texCoord;\n"
"uniform sampler2D tex;\n"

"uniform float r;\n"
"uniform float g;\n"
"uniform float b;\n"
"uniform float brightness;\n"
"uniform float contrast;\n"
"uniform float saturation;\n"

"uniform float power;\n"

"vec3 applyHue(vec3 aColor, float aHue)\n"
"{\n"
"    float angle = radians(aHue);\n"
"    vec3 k = vec3(0.57735, 0.57735, 0.57735);\n"
"    float cosAngle = cos(angle);\n"
"    return aColor * cosAngle + cross(k, aColor) * sin(angle) + k * dot(k, aColor) * (1.0 - cosAngle);\n"
"}"

"vec4 applyHSBEffect(vec4 startColor, vec4 hsbc)\n"
"{\n"
"    float _Hue = 360.0 * hsbc.r;\n"
"    float _Brightness = hsbc.g * 2.0 - 1.0;\n"
"    float _Contrast = hsbc.b * 2.0;\n"
"    float _Saturation = hsbc.a * 2.0;\n"

"    vec4 outputColor = startColor;\n"
"    outputColor.rgb = applyHue(outputColor.rgb, _Hue);\n"
"    outputColor.rgb = (outputColor.rgb - 0.5) * (_Contrast) + 0.5;\n"
"    outputColor.rgb = outputColor.rgb + _Brightness;\n"
"    float intensity = dot(outputColor.rgb, vec3(0.299,0.587,0.114));\n"
"    vec3 intensity3 = vec3(intensity,intensity,intensity);\n"
"    outputColor.rgb = mix(intensity3, outputColor.rgb, _Saturation);\n"

"    return outputColor;\n"
"}\n"

"void main() {\n"
"  vec4 diffuseColor = texture2D(tex, texCoord);\n"
"  vec4 originalColor = diffuseColor;\n"
"  vec3 sat = applyHSBEffect(diffuseColor,vec4(0, brightness,contrast,saturation)).rgb;\n"
"  vec3 gamma = vec3(r,g,b);\n"

"  diffuseColor.rgb = clamp(vec3(sat.r,sat.g,sat.b),vec3(0.0),vec3(1.0));\n"
"  diffuseColor.rgb = pow(diffuseColor.rgb, (1.0 / gamma));\n"
"  gl_FragColor.rgb = mix(gl_FragColor.rgb,diffuseColor.rgb,power);\n"
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

static void eval_secondary_expr(AVFilterContext *ctx, AVExpr *pexpr, float *value, const char *name)
{
	GLSLContext *s = ctx->priv;
	av_log(ctx, AV_LOG_VERBOSE, "eval_expr '%s'\n", name);

	*value = av_clipf(av_expr_eval(pexpr, s->var_values, NULL), 0.0, 5.0);
	av_log(ctx, AV_LOG_VERBOSE, "eval_expr '%s' end\n", name);
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

static inline int set_param(AVExpr **pexpr, const char *args, const char *cmd,
	float *value, const char name, AVFilterContext *ctx)
{
	int ret;
	GLSLContext *s = ctx->priv;

	if ((ret = set_expr(pexpr, args, cmd, ctx)) < 0)
		return ret;
	if (s->eval_mode == EVAL_MODE_INIT)
		eval_secondary_expr(ctx, *pexpr, value, name);
	
	return 0;
}

static int process_command(AVFilterContext *ctx, const char *cmd, const char *args,
                           char *res, int res_len, int flags)
{
    int ret;
	GLSLContext *s = ctx->priv;

    av_log(ctx, AV_LOG_DEBUG, "process_command cmd:%s args:%s\n",
           cmd, args);

	if (strcmp(cmd, "power") == 0)
		ret = set_expr(&s->power_pexpr, args, cmd, ctx);
	else
		ret = AVERROR(ENOSYS);

#define SET_PARAM(param_name, value_prop_name)                              \
    if (!strcmp(cmd, #param_name)) return set_param(&s->param_name##_pexpr, args, cmd, &s->value_prop_name, #param_name, ctx);

	SET_PARAM(brightness, brightness)
	else SET_PARAM(saturation, saturation)
	else SET_PARAM(contrast, contrast)
	else SET_PARAM(r, adjust_r)
	else SET_PARAM(g, adjust_g)
	else SET_PARAM(b, adjust_b)
	else return AVERROR(ENOSYS);

    if (ret < 0)
        return ret;

    if (s->eval_mode == EVAL_MODE_INIT) {
        eval_expr(ctx);
        av_log(ctx, AV_LOG_VERBOSE, "pow:%f powi:%f\n",
               s->var_values[VAR_POWER], s->power);
    }
    av_log(ctx, AV_LOG_DEBUG, "process_command end\n");
    return ret;
}

static int load_textfile(AVFilterContext *ctx, char *textfile, char **text)
{
	unsigned long fsize;
	FILE *f;

	av_log(ctx, AV_LOG_VERBOSE,
		"load_textfile '%s'\n",
		textfile);

	f = fopen(textfile, "rb");

	if (!f) {
		av_log(ctx, AV_LOG_ERROR, "invalid transition source file \"%s\"\n", textfile);
		return -1;
	}

	av_log(ctx, AV_LOG_DEBUG, "load_textfile 2\n");

	fseek(f, 0, SEEK_END);
	fsize = ftell(f);
	av_log(ctx, AV_LOG_DEBUG, "load_textfile 3 %d\n", fsize);
	fseek(f, 0, SEEK_SET);

	*text = malloc(fsize + 1);
	av_log(ctx, AV_LOG_DEBUG, "load_textfile 4\n");
	fread(*text, fsize, 1, f);
	av_log(ctx, AV_LOG_DEBUG, "load_textfile 5\n");
	fclose(f);
	av_log(ctx, AV_LOG_DEBUG, "load_textfile 6\n");

	(*text)[fsize] = 0;
	av_log(ctx, AV_LOG_DEBUG, "load_textfile 7\n");

    return 0;
}

static GLuint build_shader(AVFilterContext *ctx, const GLchar *shader_source, GLenum type) {
	GLint logSize, status;
	GLchar *errorLog;
    GLuint shader;

    av_log(ctx, AV_LOG_VERBOSE, "build_shader\n");
	shader = glCreateShader(type);
    if (!shader || !glIsShader(shader)) {
        return 0;
    }
    glShaderSource(shader, 1, &shader_source, 0);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

	if (status == GL_TRUE) {
		av_log(ctx, AV_LOG_VERBOSE, "build_shader end %d\n", status);
		return shader;
	}
	else {
		logSize = 0;
		av_log(ctx, AV_LOG_ERROR, "build_shader failed %d\n", status);
		av_log(ctx, AV_LOG_VERBOSE, "%s\n", shader_source);
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
		errorLog = (GLchar*)malloc(logSize);
		glGetShaderInfoLog(shader, logSize, &logSize, errorLog);
		av_log(ctx, AV_LOG_VERBOSE, "build_shader compilation error %s\n", errorLog);
		glDeleteShader(shader); // Don't leak the shader.
		return 0;
	}
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

static void setup_uniforms(AVFilterLink *fromLink) {
    AVFilterContext *ctx = fromLink->dst;
    GLSLContext *c = ctx->priv;

    glUniform1f(glGetUniformLocation(c->program, "power"), 0.0f);

    if (c->shader == SHADER_TYPE_MATRIX) {
        glUniform1f(glGetUniformLocation(c->program, "time"), 0.0f);
        glUniform1f(glGetUniformLocation(c->program, "dropSize"), 0.0f);
    } else if (c->shader == SHADER_TYPE_SHOCKWAVE) {
        glUniform1f(glGetUniformLocation(c->program, "time"), 0.0f);
    } else if (c->shader == SHADER_TYPE_VINTAGE) {
        glUniform1f(glGetUniformLocation(c->program, "time"), 0.0f);
        glUniform1i(glGetUniformLocation(c->program, "isColor"), 0);
    } else if (c->shader == SHADER_TYPE_ADJUST) {
        glUniform1f(glGetUniformLocation(c->program, "r"), 0.0f);
        glUniform1f(glGetUniformLocation(c->program, "g"), 0.0f);
        glUniform1f(glGetUniformLocation(c->program, "b"), 0.0f);
        glUniform1f(glGetUniformLocation(c->program, "brightness"), 0.0f);
        glUniform1f(glGetUniformLocation(c->program, "contrast"), 0.0f);
        glUniform1f(glGetUniformLocation(c->program, "saturation"), 0.0f);
    }
    //else if (c->shader == SHADER_TYPE_TRANSITION) {

    //}
}

static int build_program(AVFilterContext *ctx) {
    GLint status;
    GLuint v_shader, f_shader;
	GLSLContext *c = ctx->priv;

    av_log(ctx, AV_LOG_VERBOSE, "build_program %d\n", c->shader);

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
        else if (c->shader == SHADER_TYPE_OLD_FILM) {
            f_shader = build_shader(ctx, f_old_film_shader_source, GL_FRAGMENT_SHADER);
        }
        else if (c->shader == SHADER_TYPE_VINTAGE) {
			f_shader = build_shader(ctx, f_vintage_shader_source, GL_FRAGMENT_SHADER);
		}
		else if (c->shader == SHADER_TYPE_ADJUST) {
			f_shader = build_shader(ctx, f_adjust_shader_source, GL_FRAGMENT_SHADER);
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
	GLSLContext *c;
	AVFilterLink *fromLink, *toLink, *outLink;
	AVFrame *outFrame;

	av_log(ctx, AV_LOG_DEBUG, "apply_transition\n");
	
	c = ctx->priv;
	fromLink = ctx->inputs[FROM];
	toLink = ctx->inputs[TO];
	outLink = ctx->outputs[0];

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
	char *transition_function;

	c = ctx->priv;
	av_log(ctx, AV_LOG_VERBOSE, "init_transition %d\n", c->shader);
	c->fs.on_event = blend_frame;
	c->first_pts = AV_NOPTS_VALUE;

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
		av_log(ctx, AV_LOG_ERROR, "failed allocation f_shader_source\n");
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
	GLSLContext *c;

	av_log(ctx, AV_LOG_DEBUG, "uninit_transition\n");
	c = ctx->priv;
	ff_framesync_uninit(&c->fs);
	av_log(ctx, AV_LOG_DEBUG, "uninit_transition 2\n");

	if (c->window) {
		av_log(ctx, AV_LOG_DEBUG, "uninit_transition 3\n");
		glDeleteTextures(1, &c->uFrom);
		glDeleteTextures(1, &c->uTo);
		glDeleteBuffers(1, &c->pos_buf);
		glDeleteProgram(c->program);
		glfwDestroyWindow(c->window);
		av_log(ctx, AV_LOG_DEBUG, "uninit_transition 4\n");
	}

	if (c->f_shader_source) {
		av_log(ctx, AV_LOG_DEBUG, "uninit_transition 5\n");
		av_freep(&c->f_shader_source);
	}
	av_log(ctx, AV_LOG_DEBUG, "uninit_transition end\n");
}

static int config_input_props(AVFilterLink *inlink) {
	int ret;
	AVFilterContext     *ctx;
	GLSLContext *c;

	ctx = inlink->dst;
	c = ctx->priv;

	av_log(ctx, AV_LOG_DEBUG, "config_input_props\n");
	//glfw
	glfwWindowHint(GLFW_VISIBLE, 0);
	av_log(ctx, AV_LOG_DEBUG, "config_input_props 2\n");
	c->window = glfwCreateWindow(inlink->w, inlink->h, "", NULL, NULL);
	av_log(ctx, AV_LOG_DEBUG, "config_input_props 3\n");
	if (!c->window) {
		av_log(ctx, AV_LOG_ERROR, "setup_gl ERROR");
		return -1;
	}
	av_log(ctx, AV_LOG_DEBUG, "config_input_props 4\n");
	glfwMakeContextCurrent(c->window);
	av_log(ctx, AV_LOG_DEBUG, "config_input_props 5\n");

#ifndef __APPLE__
	glewExperimental = GL_TRUE;
	glewInit();
#endif

	av_log(ctx, AV_LOG_DEBUG, "config_input_props 6\n");
	glViewport(0, 0, inlink->w, inlink->h);

	av_log(ctx, AV_LOG_DEBUG, "config_input_props 7\n");
	if ((ret = build_program(ctx)) < 0) {
		return ret;
	}
	av_log(ctx, AV_LOG_DEBUG, "config_input_props 8\n");
	c->var_values[VAR_MAIN_W] = c->var_values[VAR_MW] = ctx->inputs[MAIN]->w;
	c->var_values[VAR_MAIN_H] = c->var_values[VAR_MH] = ctx->inputs[MAIN]->h;
	c->var_values[VAR_POWER] = NAN;
	c->var_values[VAR_T] = NAN;
	c->var_values[VAR_POS] = NAN;

	if (c->shader == SHADER_TYPE_TRANSITION) {
		av_log(ctx, AV_LOG_VERBOSE,
			"from w:%d h:%d to w:%d h:%d\n",
			ctx->inputs[FROM]->w, ctx->inputs[FROM]->h,
			ctx->inputs[TO]->w, ctx->inputs[TO]->h);
	}
	else {
		if (c->shader == SHADER_TYPE_ADJUST) {
			if ((ret = set_expr(&c->contrast_pexpr, c->contrast_expr, "contrast", ctx)) < 0 ||
				(ret = set_expr(&c->brightness_pexpr, c->brightness_expr, "brightness", ctx)) < 0 ||
				(ret = set_expr(&c->saturation_pexpr, c->saturation_expr, "saturation", ctx)) < 0 ||
				(ret = set_expr(&c->r_pexpr, c->r_expr, "r", ctx)) < 0 ||
				(ret = set_expr(&c->g_pexpr, c->g_expr, "g", ctx)) < 0 ||
				(ret = set_expr(&c->b_pexpr, c->b_expr, "b", ctx)) < 0 ||
				(ret = set_expr(&c->power_pexpr, c->power_expr, "power", ctx)) < 0)
				return ret;
		} else {
			if ((ret = set_expr(&c->power_pexpr, c->power_expr, "power", ctx)) < 0)
				return ret;
		}

		if (c->eval_mode == EVAL_MODE_INIT) {
			eval_expr(ctx);
			av_log(ctx, AV_LOG_INFO, "pow:%f powi:%f\n",
				c->var_values[VAR_POWER], c->power);
			if (c->shader == SHADER_TYPE_ADJUST) {
				eval_secondary_expr(ctx, c->r_pexpr, &(c->adjust_r), "r");
				eval_secondary_expr(ctx, c->g_pexpr, &(c->adjust_g), "g");
				eval_secondary_expr(ctx, c->b_pexpr, &(c->adjust_b), "b");

				eval_secondary_expr(ctx, c->brightness_pexpr, &(c->brightness), "brightness");
				eval_secondary_expr(ctx, c->contrast_pexpr,	  &(c->contrast),	"contrast");
				eval_secondary_expr(ctx, c->saturation_pexpr, &(c->saturation), "saturation");

				av_log(ctx, AV_LOG_INFO, "rgb:%f,%f,%f bcs: %f,%f,%f \n",
					c->adjust_r, c->adjust_g, c->adjust_b,
					c->brightness, c->contrast, c->saturation);
			}
		}
	}


	glUseProgram(c->program);
	setup_vbo(c, ctx);
	setup_uniforms(inlink);
	setup_tex(inlink);
	av_log(ctx, AV_LOG_DEBUG, "config_input_props end\n");

	return 0;
}

static int config_transition_output(AVFilterLink *outLink)
{
	AVFilterContext *ctx;
	GLSLContext *c;
	AVFilterLink *fromLink, *toLink;

	ctx = outLink->src;
	av_log(ctx, AV_LOG_DEBUG, "config_transition_output\n");
	c = ctx->priv;
	fromLink = ctx->inputs[FROM];
	toLink = ctx->inputs[TO];
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

	ret = ff_framesync_configure(&c->fs);
	av_log(ctx, AV_LOG_DEBUG, "config_transition_output end %d\n", ret);
	return ret;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in) {
	AVFilterContext *ctx;
	AVFilterLink    *outlink;
	GLSLContext *c;
	AVFrame *out;
	int skipRender;
    
	ctx     = inlink->dst;
    outlink = ctx->outputs[0];
    c = ctx->priv;
    skipRender = 0;

    av_log(ctx, AV_LOG_VERBOSE, "filter_frame\n");

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);
	glfwMakeContextCurrent(c->window);
	glUseProgram(c->program);

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

	  if (c->shader == SHADER_TYPE_ADJUST) {
		  eval_secondary_expr(ctx, c->r_pexpr, &(c->adjust_r), "r");
		  eval_secondary_expr(ctx, c->g_pexpr, &(c->adjust_g), "g");
		  eval_secondary_expr(ctx, c->b_pexpr, &(c->adjust_b), "b");

		  eval_secondary_expr(ctx, c->brightness_pexpr, &(c->brightness), "brightness");
		  eval_secondary_expr(ctx, c->contrast_pexpr, &(c->contrast), "contrast");
		  eval_secondary_expr(ctx, c->saturation_pexpr, &(c->saturation), "saturation");

		  av_log(ctx, AV_LOG_VERBOSE, "rgb:%f,%f,%f bcs: %f,%f,%f \n",
			  c->adjust_r, c->adjust_g, c->adjust_b,
			  c->brightness, c->contrast, c->saturation);
	  }
    }
    if (c->power == 0.0){
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame skip render because power is 0\n");
        skipRender = 1;
    }

    if (c->shader == SHADER_TYPE_MATRIX){
        GLfloat time = (GLfloat)(c->var_values[VAR_T] == NAN? c->frame_idx * 330: c->var_values[VAR_T] * 1000);
        GLfloat dropSize = (GLfloat)c->dropSize;
        
		av_log(ctx, AV_LOG_VERBOSE, "filter_frame matrix time:%f dropSize:%f\n", time, dropSize);
		
		glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &c->power);
		glUniform1fv(glGetUniformLocation(c->program, "time"), 1, &time);
        glUniform1fv(glGetUniformLocation(c->program, "dropSize"), 1, &dropSize);
    } else if (c->shader == SHADER_TYPE_SHOCKWAVE){
        GLfloat time = (GLfloat)(c->var_values[VAR_T] == NAN? c->frame_idx * 1.6667: c->var_values[VAR_T] * 5);
        
		av_log(ctx, AV_LOG_VERBOSE, "filter_frame shockwave time:%f\n", time);
		
		glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &c->power);
		glUniform1fv(glGetUniformLocation(c->program, "time"), 1, &time);
    } else if (c->shader == SHADER_TYPE_VINTAGE){
        GLfloat time = (GLfloat)(c->var_values[VAR_T] == NAN? c->frame_idx * 330: c->var_values[VAR_T] * 1000);
        GLint isColor = c->is_color == IS_COLOR_MODE_TRUE? 1: 0;
        
		av_log(ctx, AV_LOG_VERBOSE, "filter_frame vintage isColor:%d time:%f\n", isColor, time);
		
		glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &c->power);
		glUniform1fv(glGetUniformLocation(c->program, "time"), 1, &time);
        glUniform1iv(glGetUniformLocation(c->program, "isColor"), 1, &isColor);
    } else if (c->shader == SHADER_TYPE_OLD_FILM){
        av_log(ctx, AV_LOG_VERBOSE, "filter_frame old film\n");

        glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &(c->power));
    } else if (c->shader == SHADER_TYPE_ADJUST) {
		av_log(ctx, AV_LOG_VERBOSE, "filter_frame adjust\n");
		if (c->adjust_r == 1.0 &&
            c->adjust_g == 1.0 &&
            c->adjust_b == 1.0 &&
            c->brightness == 0.5 &&
            c->saturation == 0.5 &&
            c->contrast == 0.5){
            av_log(ctx, AV_LOG_VERBOSE, "filter_frame skip render because all values are default\n");
            skipRender = 1;
        } else {
            glUniform1fv(glGetUniformLocation(c->program, "power"), 1, &(c->power));
            glUniform1fv(glGetUniformLocation(c->program, "r"), 1, &(c->adjust_r));
            glUniform1fv(glGetUniformLocation(c->program, "g"), 1, &(c->adjust_g));
            glUniform1fv(glGetUniformLocation(c->program, "b"), 1, &(c->adjust_b));
            glUniform1fv(glGetUniformLocation(c->program, "brightness"), 1, &(c->brightness));
            glUniform1fv(glGetUniformLocation(c->program, "contrast"),	 1, &(c->contrast));
            glUniform1fv(glGetUniformLocation(c->program, "saturation"), 1, &(c->saturation));
		}
	}
    if (skipRender == 0) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inlink->w, inlink->h, 0, PIXEL_FORMAT, GL_UNSIGNED_BYTE, in->data[0]);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glReadPixels(0, 0, outlink->w, outlink->h, PIXEL_FORMAT, GL_UNSIGNED_BYTE, (GLvoid *) out->data[0]);
        av_frame_free(&in);
    } else {
        out = in;
    }

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

#define FREE_PARAM(param_name)                              \
	if (c->param_name##_pexpr){								\
		av_expr_free(c->param_name##_pexpr);				\
		av_log(ctx, AV_LOG_VERBOSE, "uninit5_%s\n", #param_name);			\
		c->param_name##_pexpr = NULL;						\
	}

	FREE_PARAM(power)
	FREE_PARAM(r)
	FREE_PARAM(g)
	FREE_PARAM(b)
	FREE_PARAM(brightness)
	FREE_PARAM(contrast)
	FREE_PARAM(saturation)
    
    av_log(ctx, AV_LOG_VERBOSE, "uninit6\n");
}

//necessary for transition only because of the f-sync
static int activate_transition(AVFilterContext *ctx)
{
	GLSLContext *c;
	
	av_log(ctx, AV_LOG_DEBUG, "activate_transition\n");
	c = ctx->priv;
	
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
			 { "adjust", "set vintage like effect", 0, AV_OPT_TYPE_CONST, {.i64 = SHADER_TYPE_ADJUST},.flags = FLAGS,.unit = "shader" },
             { "old_film", "set old-film like effect", 0, AV_OPT_TYPE_CONST, {.i64 = SHADER_TYPE_OLD_FILM},.flags = FLAGS,.unit = "shader" },
			 { "none", "passthrough", 0, AV_OPT_TYPE_CONST, {.i64 = SHADER_TYPE_PASSTHROUGH},.flags = FLAGS,.unit = "shader" },
	{ "vs_textfile",    "set a text file for vertex shader",        OFFSET(vs_textfile),           AV_OPT_TYPE_STRING, {.str=NULL},  CHAR_MIN, CHAR_MAX, FLAGS},
    { "fs_textfile",    "set a text file for fragment shader",      OFFSET(fs_textfile),           AV_OPT_TYPE_STRING, {.str=NULL},  CHAR_MIN, CHAR_MAX, FLAGS},
	{ "power", "set the power expression", OFFSET(power_expr), AV_OPT_TYPE_STRING, {.str = "0"}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "eval", "specify when to evaluate expressions", OFFSET(eval_mode), AV_OPT_TYPE_INT, {.i64 = EVAL_MODE_FRAME}, 0, EVAL_MODE_NB-1, FLAGS, "eval" },
             { "init",  "eval expressions once during initialization", 0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_INIT},  .flags = FLAGS, .unit = "eval" },
             { "frame", "eval expressions per-frame",                  0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_FRAME}, .flags = FLAGS, .unit = "eval" },
	{ "r", "set the r expression", OFFSET(r_expr), AV_OPT_TYPE_STRING, {.str = "1"}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "g", "set the g expression", OFFSET(g_expr), AV_OPT_TYPE_STRING, {.str = "1"}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "b", "set the b expression", OFFSET(b_expr), AV_OPT_TYPE_STRING, {.str = "1"}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "brightness", "set the brightness expression", OFFSET(brightness_expr), AV_OPT_TYPE_STRING, {.str = "0.5"}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "contrast", "set the contrast expression", OFFSET(contrast_expr), AV_OPT_TYPE_STRING, {.str = "0.5"}, CHAR_MIN, CHAR_MAX, FLAGS },
	{ "saturation", "set the saturation expression", OFFSET(saturation_expr), AV_OPT_TYPE_STRING, {.str = "0.5"}, CHAR_MIN, CHAR_MAX, FLAGS },
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