Colab data and code

1) A100 40GB 
2) measure batch size as function of throughput and display KV cache size per run. Does LV Cache size stay constant? 

3) next step change colab workbook into dataclass and design a test harness around vllm, the batch size parameters and save the runs so we dont have to rerun this configuration 
4) create a react app which displays the a100 40GB colab batch size/throughput graphs. Place the react under llm_benchmark/simple_dashboard
