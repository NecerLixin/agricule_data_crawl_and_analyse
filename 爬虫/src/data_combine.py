import pandas as pd

if __name__ == '__main__':
    data_list = []
    for i in range(7):
        file_path = f'../data/data{str(i)}.xlsx'
        temp_data = pd.read_excel(file_path)
        data_list.append(temp_data)
    data_all = pd.concat(data_list)
    file = '../data/data.xlsx'
    data_all.to_excel(file,sheet_name='1',index=False)
