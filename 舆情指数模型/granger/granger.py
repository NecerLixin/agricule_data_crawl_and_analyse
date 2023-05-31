from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

def get_data(file_path):
    data = pd.read_excel(file_path)
    return data
    
if __name__ == "__main__":
    data = get_data('granger/granger_price_prob.xlsx')
    # grangercausalitytests(data[['价格波动','舆情指数']],maxlag=2)
    # grangercausalitytests(data[['柑橘价格','舆情指数']],maxlag=2)
    # grangercausalitytests(data[['柑橘价格','舆情指数']],maxlag=2)
    # data['舆情指数'] -= 0.5
    grangercausalitytests(data[['舆情指数','柑橘价格']],maxlag=2)
    