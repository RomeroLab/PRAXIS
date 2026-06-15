"""Compatibility shim for proteinnpt against recent transformers.

proteinnpt (1.5.1) does ``from transformers import ConvBertConfig, ConvBertLayer``,
but recent transformers no longer exports ``ConvBertLayer`` at the top level (it
lives in ``transformers.models.convbert.modeling_convbert``). proteinnpt's package
``__init__`` eagerly imports the offending module, so importing anything from
proteinnpt fails with an ImportError.

Import this module BEFORE importing proteinnpt; it re-exports ``ConvBertLayer`` onto
the transformers namespace so the legacy import resolves. Idempotent and a no-op on
transformers versions that still export the symbol.
"""
import transformers

if not hasattr(transformers, "ConvBertLayer"):
    from transformers.models.convbert.modeling_convbert import ConvBertLayer

    transformers.ConvBertLayer = ConvBertLayer
