# QuantBench

This project serves as a benchmark for evaluating AI methods for quantitative stock investments. We cover temporal models, graph neural networks, and end-to-end reinforcement learning models for a comprehensive evaluation.

## Quickstart

Install `q4l` before running this benchmark

```bash
pip install git+https://github.com/SaizhuoWang/q4l.git@main
```

Enjoy!

## Configurations

This benchmark is built upon `q4l`, which is a python project for quant research. `q4l` applies `hydra` for configuration management. In short, we use a structured config schema for configuration, and the structure is included in `q4l/config/__init__.py`.

Please refer to [hydra documentation](https://hydra.cc/docs/intro) for hydra help.

## Graph neural networks

Running GNN experiments requires the following steps:F

1. **Get data**: Volume-price data is handled by `q4l`. You don't have to worry about it. Graph data is provided in some raw format, and you should read it into memory by yourself. Please write relevant codes in your model file.
2. **Build model**: We use `lightning` to structure our model. `lightning` provides useful templates for building models. It is suggested to write your graph loading, model construction, training, validation and prediction models as methods for your models. Please refer to [lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/) for help.
3. **Train model**: After you have successfully constructed the model, training is very easy. You just instantiate a `Trainer` and call `trainer.fit(model)`. Please refer to [lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/) for help.
4. **Evaluate model**: Evaluation protocol is builtin. You don't have to worry about it.

### Load graph data

- **Wikidata graph**: We constructed a stock-subgraph from Wikidata. The processed data is stored at `data/wikidata/stock_graph`. Sub-directories under this dir represents region (e.g. `us`).
You may refer to [`q4l/model/zoo/spatial/base.py`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/model/zoo/spatial/fundamental/base.py#L31) for data loading.
Current data contains 1-hop and 2-hop stock relations. Specifically:
  - `stock_records.jsonl`: Each line corresponds to a stock. There might be duplicates.
  - `intermediate_nodes.jsonl`: Each line corresponds to an entity in Wikidata that links two stock entities.
  - `intra_stock_relations.jsonl`: Each line corresponds to a relation between two stocks.

- **Industry graph**: `data/industry/us.csv` contains the industry categorization for US stocks. The `WindCodes` column corresponds to ticker code (but with some suffix such as ".N"), and the `INDUSTRY_GICS` column corresponds to the industry category.

### DGL-representation of stock graph

Graph-structured data is parsed by their corresponding helper functions into `data_dicts`, and these different graph information is combined together to form a large `DGLGraph`. This graph is attached under the [`StockKG`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/model/zoo/spatial/fundamental/base.py#L117) class. This heterograph has only 1 node type ('stock') and various other edge types.

At each cross-section, you may need to pick a sub-graph from this whole big graph based on node information (stock feature alignment) or edge information (split into multiple graphs like in HATR). These features are provided in `StockKG` via `get_node_subgraph` and `get_info_subgraph`.

### Developing new GNNs

Your GNN should instantiate a `StockKG` object at init, and transform it into your own forms at each iteration.
A class inheritance chain is [`QuantModel`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/model/base.py#L35) -> [`SpatiotemporalModel`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/model/base.py#L221) -> [`KGModel`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/model/zoo/spatial/fundamental/base.py#L167) -> [`YourOwnModel (e.g. RSR)`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/model/zoo/spatial/fundamental/rsr.py)

### Test your model

After you have implemented your model, you should compose a dedicated config file for it. For example, after I have implemented RSR, I need to compose the file [`q4l/config/q4l_exp_default/experiment/model/rsr.yaml`](http://gitlab.finai.idea.edu.cn/wangsaizhuo/wszlib/-/blob/sub-master/q4l/config/q4l_builtin/experiment/model/rsr.yaml). Then, to test my model, I can run the following command:

```bash
python main.py -cn benchmark \
  job/machine=dgx \
  experiment/model=rsr
```

### Debugging

I know that current code may seem very involved and complicated. And if you want to walk it through, you may debug into the `main.py` where everything starts. Generally speaking:

1. The experiment is an `ensemble-rolling` experiment. `ensemble` means it is repeated for several times, and `rolling` means for a test period, e.g. consisting of 1000 trading days, we may split it into 10 segments where each segment contains 100 trading days. In this way an atomic run is essentially 1 rolling segment in 1 of the ensemble runs. So the outer loop of the main script is the ensemble loop, and the inner loop is the rolling loop. Inside each rolling loop, the data and model training parts are involved.

2. Data loading is tedious, and you don't need to worry about it. Just keep in mind that for GNN model, each data sample (batch size=1) returns all stocks in the stock pool for a specific trading day.

3. You should just focus on model training. For an example implementation, please refer to `q4l.model.base.TimeSeriesModel`. Specifically, `trainer.fit(model, data)` will involve training and validation, while `trainer.predict(model, data)` will run inference and dump the results to disk via `mlflow`.

4. After run has finished, evaluation is automatically run, and you don't need to worry about it either.
