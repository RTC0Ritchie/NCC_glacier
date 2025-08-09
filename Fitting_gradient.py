import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import argparse

#%%

class multiFitting:
    def __init__(self, d, is_given_a=False, a=None, is_given_dv=False,dv=None):
        self.d = d
        self.is_given_a = is_given_a
        if self.is_given_a:
            self.a = a
        self.is_given_dv2 = is_given_dv
        if self.is_given_dv2:
            self.dv2 = dv**2
        self.results=[]
        if self.is_given_a and self.a is None:
            raise ValueError("a is not given")
        if self.is_given_dv2 and self.dv2 is None:
            raise ValueError("dv is not given")

    def func_00(self,x,k,dv2):
        return np.sqrt(abs(k)/(x**2) + dv2)
    def func_10(self,x,dv2):
        return np.sqrt(self.a*(self.d**2)/(x**2) + dv2)
    def func_01(self,x,k):
        return np.sqrt(abs(k)/(x**2) + self.dv2**2)

    def fit_process(self,data_dt, data_dv2):
        if not self.is_given_a and not self.is_given_dv2:
            popt, _ = curve_fit(self.func_00, data_dt, data_dv2)
            fit_paras = np.array([popt[0],popt[1]])
        if self.is_given_a and not self.is_given_dv2:
            popt, _ = curve_fit(self.func_10, data_dt, data_dv2)
            fit_paras = np.array([self.a*self.d**2,popt[0]])
        if not self.is_given_a and self.is_given_dv2:
            popt, _ = curve_fit(self.func_01, data_dt, data_dv2)
            fit_paras = np.array([popt[0],self.dv2])
        if self.is_given_a and self.is_given_dv2:
            fit_paras = np.array([self.a*self.d**2,self.dv2])
        if fit_paras[1] < 0:
            print('Warning: dv^2 is negative')
        self.results = [abs(fit_paras[0])/(self.d**2), np.sqrt(fit_paras[1])]

def parse_txt_config(path):
    cfg = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            # bool / int / float / list
            if val.lower() in ('true', 'false'):
                cfg[key] = val.lower() == 'true'
            else:
                parts = val.split()
                try:
                    if len(parts) > 1:
                        cfg[key] = [int(p) if p.isdigit() else float(p) for p in parts]
                    else:
                        cfg[key] = int(val) if val.isdigit() else float(val)
                except ValueError:
                    cfg[key] = val
    return cfg
def parse_args():
    parser = argparse.ArgumentParser(description='Line Gradient via CLI or txt config.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', type=str,
                       help='Path to txt config file (key = value per line)')
    # The remaining parameters remain compatible and only take effect without --config
    parser.add_argument('--d', type=float)
    parser.add_argument('--is_given_a', type=bool)
    parser.add_argument('--a_given', type=float)
    parser.add_argument('--is_given_dv', type=bool)
    parser.add_argument('--dv', type=float)
    parser.add_argument('--gra_sta', type=str)
    parser.add_argument('--gra_sta_row', type=int)
    return parser.parse_args()
def main():
    args = parse_args()
    if args.config:
        cfg = parse_txt_config(args.config)
    else:
        cfg = vars(args)

    def get(k, default=None):
        return cfg.get(k, default)

    d = float(get('d'))
    is_given_a = bool(get('is_given_a'))
    a_given = float(get('a_given'))
    is_given_dv = bool(get('is_given_dv'))
    dv_given = float(get('dv'))
    gra_sta = str(get('gra_sta'))
    gra_sta_row = int(get('gra_sta_row'))

    main_data = np.array(pd.read_csv(gra_sta+'.csv'))
    data_dt = main_data[:, 2]
    data_dv2 = main_data[:, gra_sta_row]

    test1 = multiFitting(d, is_given_a=is_given_a, a=a_given, is_given_dv=is_given_dv, dv=dv_given)
    test1.fit_process(data_dt, data_dv2)
    print(test1.results)

if __name__ == '__main__':
    main()