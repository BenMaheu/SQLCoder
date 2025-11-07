# Install dependencies
Install all dependencies using poetry:
```
poetry install
```

Poetry does not have a correct torch dependency specification yet, so please install it manually.
```
pip install torch
```

For accelerate : 
```
pip install -U "transformers[torch]"
```