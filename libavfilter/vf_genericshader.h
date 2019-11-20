#ifndef AVFILTER_GENERICSHADER_H
#define AVFILTER_GENERICSHADER_H

#include "libavutil/eval.h"
#include "libavutil/pixdesc.h"
#include "framesync.h"
#include "avfilter.h"

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

#endif /* AVFILTER_GENERICSHADER_H */