#!/usr/bin/env python3
"""
Clear_statistic.py
Usage:
    python modify_csv.py --file input.csv --mode 1    # Only clear the last row
    python modify_csv.py --file input.csv --mode 2    # Clear all rows
"""


import argparse
import pandas as pd
import os

file_name = ["vx_", "vy_", "isprocessed_"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sta', type=str, required=True)
    parser.add_argument('--mode', type=int, choices=[1, 2], required=True)
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.sta):
        print(f"Error: File {args.sta} does not exist.")
        return

    try:
        df = pd.read_csv(args.sta)
    except Exception as e:
        print(f"Error: Failed to read CSV file. {e}")
        return

    if args.mode == 1:
        last_second_column = df.shape[1] - 2
        last_one_index = df[df.iloc[:, last_second_column] == 1].index.max()
        if last_one_index is not None:
            df.iloc[last_one_index, 3] = 0
            df.iloc[last_one_index, 4] = 0
        v_str = str(int(df.iloc[last_one_index, 0])) + '-' + str(int(df.iloc[last_one_index, 1]))
        print(f"Last row cleared: {v_str}")
        err_name_urf_equal = [
            file_name[0] + v_str,
            file_name[1] + v_str,
            file_name[2] + v_str
        ]
        for err_name in err_name_urf_equal:
            file_err = args.file+'/dataf/err_'+err_name+'_f.csv'
            if os.path.exists(file_err):
                os.remove(file_err)
                print(f'{file_err} removed')
            file_err = args.file+'/datae/err_'+err_name+'_e.csv'
            if os.path.exists(file_err):
                os.remove(file_err)
                print(f'{file_err} removed')
            file_err = args.file+'/dataf/'+err_name+'_f.csv'
            if os.path.exists(file_err):
                os.remove(file_err)
                print(f'{file_err} removed')
            file_err = args.file+'/datae/'+err_name+'_e.csv'
            if os.path.exists(file_err):
                os.remove(file_err)
                print(f'{file_err} removed')
    elif args.mode == 2:
        # 模式 2：将所有行的第0列和第1列全部置零
        df.iloc[:, [3, 4]] = 0
        # 模式 2：删除所有文件
        for i in range(df.shape[0]):
            v_str = str(int(df.iloc[i, 0])) + '-' + str(int(df.iloc[i, 1]))
            err_name_urf_equal = [
                file_name[0] + v_str,
                file_name[1] + v_str,
                file_name[2] + v_str
            ]
            for err_name in err_name_urf_equal:
                file_err = args.file+'/dataf/err_'+err_name+'_f.csv'
                if os.path.exists(file_err):
                    os.remove(file_err)
                    print(f'{file_err} removed')
                file_err = args.file+'/datae/err_'+err_name+'_e.csv'
                if os.path.exists(file_err):
                    os.remove(file_err)
                    print(f'{file_err} removed')
                file_err = args.file+'/dataf/'+err_name+'_f.csv'
                if os.path.exists(file_err):
                    os.remove(file_err)
                    print(f'{file_err} removed')
                file_err = args.file+'/datae/'+err_name+'_e.csv'
                if os.path.exists(file_err):
                    os.remove(file_err)
                    print(f'{file_err} removed')

    try:
        df.to_csv(args.sta,index=False)
        print(f"Modified CSV saved to {args.sta}")
    except Exception as e:
        print(f"Error: Failed to save CSV file. {e}")

if __name__ == '__main__':
    main()