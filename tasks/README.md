### postprocessing

metric normalization

```python
content["results"]["community|gpqa-fr|0"]["norm_acc"] = max(0., (float(content["results"]["community|gpqa-fr|0"]["acc"])-0.25)/0.75) #normalization with 4 choices
content["results"]["community|ifeval-fr|0"]["norm_acc"] = ((float(content["results"]["community|ifeval-fr|0"]["prompt_level_strict_acc"])
                                                                +float(content["results"]["community|ifeval-fr|0"]["inst_level_strict_acc"]))
                                                                   /2)
content["results"]["community|sornette|0"]["norm_acc"] = max(0., (float(content["results"]["community|sornette|0"]["acc"])-0.25)/0.75) #normalization with 4 choices
content["results"]["community|kangourou-to|0"]["norm_acc"] = max(0., (float(content["results"]["community|kangourou-to|0"]["acc"])-0.2)/0.8) #normalization with 5 choices
```
