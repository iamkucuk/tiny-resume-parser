"""tiny-resume-parser: small vision-language models that read a CV page image and write ATS JSON.

Modules, in the order data flows through them:
    schema    the output JSON schema (pydantic)          label     teacher labelling (GLM-5.3-Flash)
    data      label files, train/validation split        models    model families, prompts, training target
    train     fine-tuning                                soup      weight averaging of fine-tuned models
    generate  model loading and answer generation        evaluate  answers + strict F1 on labelled pages
    metrics   strict / lenient / content F1, repair      rescore   all metrics on saved answers
    parse     CV files (images, PDFs) -> JSON
"""

__version__ = "0.1.0"
