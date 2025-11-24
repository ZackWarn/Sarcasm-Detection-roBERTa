import traceback
try:
    import transformers
    print('transformers OK:', transformers.__version__)
except Exception as e:
    traceback.print_exc()
    print('transformers import failed:', e)
