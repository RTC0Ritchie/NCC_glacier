import numpy as np
import cv2 as cv2
import pandas as pd
import os
from tqdm import tqdm
import ast
import argparse
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator

'''
The velocity is positive when moving upward and positive when moving right, with the unit being m/day
'''

# ------------------------------------------------------------------
# Global configuration (can be turned into CLI flags if needed)
# ------------------------------------------------------------------
file_name = ['vx_', 'vy_', 'isprocessed_']
subfolders = ['dataf', 'datae']

spatial_resolution = 10
is_need_interp = True
nye_maxT = 1
nye_tol = 1e-6
nye_dt0 = 0.1
# ------------------------------------------------------------------
SYS_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'System_config.txt')
def load_sys_config():
    cfg = {}
    with open(SYS_CFG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = map(str.strip, line.split('=', 1))
            try:
                cfg[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                cfg[key] = val
    return cfg
SYS_CFG = load_sys_config()
globals().update({
    'file_name': SYS_CFG['file_name'],
    'subfolders': SYS_CFG['subfolders'],
    'spatial_resolution': SYS_CFG['spatial_resolution'],
    'is_need_interp': SYS_CFG['is_need_interp'],
    'nye_maxT': SYS_CFG['nye_maxT'],
    'nye_tol': SYS_CFG['nye_tol'],
    'nye_dt0': SYS_CFG['nye_dt0']
})
SYS_CFG = load_sys_config()

def fast_linear_interpolation(arr):
    '''
    对包含 NaN 值的二维数组进行快速线性插值
    '''
    # 创建一个掩码，标记 NaN 值位置
    mask = np.isnan(arr)

    # 找到非 NaN 值的位置和值
    valid_points = ~mask
    valid_x, valid_y = np.where(valid_points)
    valid_values = arr[valid_x, valid_y]

    # 创建线性插值器
    interpolator = LinearNDInterpolator(
        np.column_stack((valid_x, valid_y)),
        valid_values
    )

    # 找到需要插值的 NaN 位置
    nan_x, nan_y = np.where(mask)

    # 插值 NaN 位置
    interpolated_values = interpolator(nan_x, nan_y)

    # 填补原始数组中的 NaN 值
    result = arr.copy()
    result[mask] = interpolated_values
    return result


class Line_gradient:
    def __init__(self,velocity_result_file,sta_name):
        self.basis_fold_urf = velocity_result_file
        self.sta_name = sta_name

        df = pd.read_csv(sta_name + '.csv')
        self.all_state_list = np.array(df)

    def which_v(self,number):
        v_last = str(self.all_state_list[number, 0])
        v_end = str(self.all_state_list[number, 1])
        v_str = v_last + '-' + v_end
        finish_name_urf = [file_name[0] + v_str , file_name[1] + v_str]

        df1 = pd.read_csv(self.basis_fold_urf +  '/' + finish_name_urf[0] + '.csv')
        df2 = pd.read_csv(self.basis_fold_urf +  '/' + finish_name_urf[1] + '.csv', )
        return np.array(df1)/spatial_resolution, -np.array(df2)/spatial_resolution, v_str

    # 计算沿线梯度-------------------------------------------------------#输入的vy必须向下为正，像素
    def get_decline_line(self, V, xrange, yrange, is_output_idx=False):
            if xrange[1] - xrange[0] > yrange[1] - yrange[0]:
                xarray = np.arange(xrange[0], xrange[1])
                yarray = yrange[0] + (yrange[1] - yrange[0]) / (xrange[1] - xrange[0]) * (xarray - xrange[0])
                varray = np.zeros(xarray.shape)
                for jj in range(xarray.shape[0]):
                    tmp = V[int(xarray[jj]), int(np.floor(yarray[jj]))] * (np.floor(yarray[jj]) + 1 - yarray[jj]) + \
                          V[int(xarray[jj]), int(np.floor(yarray[jj])) + 1] * (yarray[jj] - np.floor(yarray[jj]))
                    if np.isnan(tmp):
                        varray[jj] = np.nan
                    else:
                        varray[jj] = tmp
                if is_output_idx:
                    return varray, (yrange[1] - yrange[0]) / (xrange[1] - xrange[0]), xarray, yarray
                else:
                    return varray, (yrange[1] - yrange[0]) / (xrange[1] - xrange[0])
            else:
                yarray = np.arange(yrange[0], yrange[1])
                xarray = xrange[0] + (xrange[1] - xrange[0]) / (yrange[1] - yrange[0]) * (yarray - yrange[0])
                varray = np.zeros(yarray.shape)
                for jj in range(yarray.shape[0]):
                    tmp = V[int(np.floor(xarray[jj])), int(yarray[jj])] * (np.floor(xarray[jj]) + 1 - xarray[jj]) + \
                          V[int(np.floor(xarray[jj])) + 1, int(yarray[jj])] * (xarray[jj] - np.floor(xarray[jj]))
                    if np.isnan(tmp):
                        varray[jj] = np.nan
                    else:
                        varray[jj] = tmp
                if is_output_idx:
                    return varray, (xrange[1] - xrange[0]) / (yrange[1] - yrange[0]), xarray, yarray
                else:
                    return varray, (xrange[1] - xrange[0]) / (yrange[1] - yrange[0])

    def expand_with_linear_pad(self, arr, pad):
        Ny, Nx = arr.shape
        yg = np.arange(Ny, dtype=float)
        xg = np.arange(Nx, dtype=float)

        # 构造外扩后的网格
        yg_pad = np.arange(-pad, Ny + pad, dtype=float)
        xg_pad = np.arange(-pad, Nx + pad, dtype=float)
        Ny_pad, Nx_pad = yg_pad.size, xg_pad.size

        # 用 RegularGridInterpolator 做线性外插
        itp = RegularGridInterpolator((yg, xg), arr,
                                      bounds_error=False,
                                      fill_value=None,  # 外插
                                      method='linear')
        YP, XP = np.meshgrid(yg_pad, xg_pad, indexing='ij')
        arr_pad = itp((YP, XP)).reshape(Ny_pad, Nx_pad)
        return arr_pad, yg_pad, xg_pad

    def track_feature_upon_index_field(self, vx_itp, vy_itp,
                                       idx1,
                                       maxT=1, tol=1e-6, dt0=0.1):

        p1 = np.array(idx1, dtype=float)  # (i1, j1) → (y1, x1)

        t, dt = 0.0, dt0
        while t < maxT:
            v1 = np.squeeze(np.array([vy_itp(p1), vx_itp(p1)]))  # vx, vy 顺序

            # Euler
            p1_e = p1 + v1 * dt

            # 改进 Euler
            v1_e = np.squeeze(np.array([vy_itp(p1_e), vx_itp(p1_e)]))
            p1_ie = p1 + 0.5 * (v1 + v1_e) * dt

            err = np.linalg.norm(p1_ie - p1_e) / (np.linalg.norm(p1_ie) + 1e-12)
            if err > tol:
                dt *= 0.5
                continue

            p1 = p1_ie
            t += dt
            if err < tol / 4:
                dt = min(dt * 2.0, maxT - t)

        return p1

    def gradient_numpy(self, V, irange, jrange, is_need_interp=False):
        # if is_need_interp:
        #     idx_nan = (1-np.isnan(V)).astype(float)
        #     idx_nan[idx_nan==0] = np.nan
        #     V = pre.fast_linear_interpolation(V)
        # else:
        #     idx_nan = 1
        varr, tank = self.get_decline_line(V, irange, jrange)
        if is_need_interp:
            # V *= idx_nan
            # return np.gradient(varr)/ np.sqrt(1+tank**2)*idx_nan
            return np.gradient(varr) / np.sqrt(1 + tank ** 2)
        else:
            return np.gradient(varr) / np.sqrt(1 + tank ** 2)

    def gradient_sobel(self, V, irange, jrange, is_need_interp=False):
        if is_need_interp:
            idx_nan = (1 - np.isnan(V)).astype(float)
            idx_nan[idx_nan == 0] = np.nan
            V = fast_linear_interpolation(V)
        else:
            idx_nan = 1
        vdx = cv2.Sobel(V, -1, 1, 0, ksize=3) / 8
        vdy = -cv2.Sobel(V, -1, 0, 1, ksize=3) / 8
        if is_need_interp:
            vdx *= idx_nan
            vdy *= idx_nan
            V *= idx_nan
        thera = np.arctan2(irange[0] - irange[1], jrange[1] - jrange[0])
        vdtheta = vdx * np.cos(thera) + vdy * np.sin(thera)
        vlinegra, _ = self.get_decline_line(vdtheta, irange, jrange)
        return vlinegra

    def gradient_spectrum(self, V, irange, jrange, is_need_interp=False):
        if is_need_interp:
            idx_nan = (1 - np.isnan(V)).astype(float)
            idx_nan[idx_nan == 0] = np.nan
            V = fast_linear_interpolation(V)
        else:
            idx_nan = 1
        Ny, Nx = V.shape
        x_phys = np.linspace(jrange[0], jrange[1], Nx)
        y_phys = np.linspace(irange[0], irange[1], Ny)

        # 线性映射到 [-1, 1]
        a, b = jrange
        c, d = irange
        x_std = 2 * (x_phys - a) / (b - a) - 1
        y_std = 2 * (y_phys - c) / (d - c) - 1

        # Chebyshev 拟合 & 微分
        # 先把 y 方向当成“样本”，对每一列做 x 方向的 Chebyshev 拟合
        cx_list = [np.polynomial.chebyshev.chebfit(x_std, V[:, j], deg=Nx - 1) for j in range(Ny)]
        # 再把得到的展开系数当成“样本”，对 x 方向做 y 方向的 Chebyshev 拟合
        c = np.array([np.polynomial.chebyshev.chebfit(y_std, [cx[k] for cx in cx_list], deg=Ny - 1)
                      for k in range(Nx)])
        cx = np.polynomial.chebyshev.chebder(c, axis=1)
        cy = np.polynomial.chebyshev.chebder(c, axis=0)
        cx_pad = np.zeros_like(c);
        cx_pad[:, :-1] = cx
        cy_pad = np.zeros_like(c);
        cy_pad[:-1, :] = cy
        dVdx_std = np.polynomial.chebyshev.chebval2d(x_std, y_std, cx_pad)
        dVdy_std = np.polynomial.chebyshev.chebval2d(x_std, y_std, cy_pad)

        # 还原到物理坐标
        dVdx = dVdx_std / ((b - a) / 2)
        dVdy = dVdy_std / ((d - c) / 2)

        if is_need_interp:
            dVdx *= idx_nan
            dVdy *= idx_nan
            V *= idx_nan

        thera = np.arctan2(irange[0] - irange[1], jrange[1] - jrange[0])
        vdtheta = dVdx * np.cos(thera) + dVdy * np.sin(thera)
        vlinegra, _ = self.get_decline_line(vdtheta, irange, jrange)
        return vlinegra

    def gradient_nye(self, vx, vy, irange, jrange, is_need_interp=False, L0=10,
                     ispadding=True, padding_grid=10, maxT=1, tol=1e-6, dt0=0.1, coe_uy2vx=1 / 2):
        L0 = L0/spatial_resolution
        A_strain_matrix = np.array([
            [0, 0, 1],
            [1 / 2, 1, 1 / 2],
            [1, 0, 0],
            [1 / 2, -1, 1 / 2]
        ])
        retrieve_matrix = np.dot(np.linalg.inv(np.dot(A_strain_matrix.T, A_strain_matrix)), A_strain_matrix.T)
        if is_need_interp:
            idx_nan = (1 - np.isnan(vx)).astype(float)
            idx_nan[idx_nan == 0] = np.nan
            vx = fast_linear_interpolation(vx)
            vy = fast_linear_interpolation(vy)

        if ispadding:
            vx_pad, yg, xg = self.expand_with_linear_pad(vx, padding_grid)
            vy_pad, _, _ = self.expand_with_linear_pad(vy, padding_grid)
        else:
            Ny, Nx = vy.shape
            xg = np.arange(Nx, dtype=float)
            yg = np.arange(Ny, dtype=float)
            vx_pad, vy_pad = vx.copy(), vy.copy()

        vx_itp = RegularGridInterpolator((yg, xg), vx_pad, bounds_error=False, fill_value=np.nan)
        vy_itp = RegularGridInterpolator((yg, xg), vy_pad, bounds_error=False, fill_value=np.nan)
        _, _, ibins, jbins = self.get_decline_line(vx, irange, jrange, is_output_idx=True)
        epsilon_c = np.zeros((ibins.shape[0], 4))  # 0,45,90,135
        for iter in range(ibins.shape[0]):
            center_i, center_j = ibins[iter], jbins[iter]
            points = [
                [center_i, center_j], [center_i - L0, center_j], [center_i, center_j + L0],
                [center_i + L0, center_j], [center_i, center_j - L0]
            ]  # 中，上，右，下，左
            newpoints = []
            for epoch in range(5):
                newpoints.append(self.track_feature_upon_index_field(vx_itp, vy_itp, points[epoch], maxT, tol, dt0))
            norm_0 = np.array([L0, np.sqrt(2)*L0, L0, np.sqrt(2)*L0])
            norm_f = [
                [np.linalg.norm(newpoints[0] - newpoints[1]), np.linalg.norm(newpoints[0] - newpoints[3])],
                [np.linalg.norm(newpoints[1] - newpoints[4]), np.linalg.norm(newpoints[2] - newpoints[3])],
                [np.linalg.norm(newpoints[0] - newpoints[2]), np.linalg.norm(newpoints[0] - newpoints[4])],
                [np.linalg.norm(newpoints[1] - newpoints[2]), np.linalg.norm(newpoints[3] - newpoints[4])]
            ]
            epsilon_c[iter, :] = np.nanmean(np.log(np.array(norm_f) / norm_0[:, np.newaxis]), axis=1) / maxT
            # print(epsilon_c[iter,:])
        epsilon_0 = np.dot(retrieve_matrix, epsilon_c.T).T

        exx, exy, eyy = epsilon_0[:, 0], epsilon_0[:, 1], epsilon_0[:, 2]
        dudx = exx;
        dvdy = eyy
        dudy = exy * coe_uy2vx;
        dvdx = exy * (1 - coe_uy2vx)
        if is_need_interp:
            # dudx *= idx_nan; dudy *= idx_nan
            # dvdy *= idx_nan; dvdx *= idx_nan
            vx *= idx_nan;
            vy *= idx_nan
        thera = np.arctan2(irange[0] - irange[1], jrange[1] - jrange[0])
        vxdtheta = dudx * np.cos(thera) + dudy * np.sin(thera)
        vydtheta = dvdx * np.cos(thera) + dvdy * np.sin(thera)
        # vxlinegra,_ = self.get_decline_line(vxdtheta,irange,jrange)
        # vylinegra,_ = self.get_decline_line(vydtheta,irange,jrange)
        return vxdtheta, vydtheta

    def process_line_gridient(self, vx, vy, irange, jrange, is_need_interp=False, GRAtype='np', nyeparas=None):

        if GRAtype == 'nye':
            if nyeparas is None:
                return self.gradient_nye(vx, vy, irange, jrange, is_need_interp)
            else:
                return self.gradient_nye(vx, vy, irange, jrange, is_need_interp, *nyeparas)
        elif GRAtype == 'sobel':
            return self.gradient_sobel(vx, irange, jrange, is_need_interp), self.gradient_sobel(-vy, irange, jrange,
                                                                                                is_need_interp)
        elif GRAtype == 'spec':
            return self.gradient_spectrum(vx, irange, jrange, is_need_interp), self.gradient_spectrum(-vy, irange,
                                                                                                      jrange,
                                                                                                      is_need_interp)
        else:
            return self.gradient_numpy(vx, irange, jrange, is_need_interp), self.gradient_numpy(-vy, irange, jrange,
                                                                                                is_need_interp)

    def auto_middle_save(self, V, name_urf):
        df = pd.DataFrame(V)
        df.to_csv(name_urf + '.csv', index=False)

    def process_line(self, target, irange, jrange, GRAtype='np', nyeparas=None,stride=1):
        bias_statis = np.array([[]])
        dayidx = 2
        for ii in tqdm(range(0, self.all_state_list.shape[0])):
            vx_l, vy_l, vstr = self.which_v(ii)

            vx_line_g, vy_line_g = self.process_line_gridient(
                vx_l, vy_l, irange, jrange,
                is_need_interp=is_need_interp, GRAtype=GRAtype, nyeparas=nyeparas
            )
            # vx_line_g_mean[jj] = np.sqrt(np.nanmean(vx_line_g**2))
            # vy_line_g_mean[jj] = np.sqrt(np.nanmean(vy_line_g**2))
            vx_line_g_mean = np.nanmean(np.abs(vx_line_g))/stride
            vy_line_g_mean = np.nanmean(np.abs(vy_line_g))/stride

            tmp = np.array([[int(vstr[:8]), int(vstr[-8:]), self.all_state_list[ii, dayidx]]])
            tmp = np.hstack((tmp, np.array([[vx_line_g_mean, vy_line_g_mean]])))
            if bias_statis.shape[1] == 0:
                bias_statis = tmp
            else:
                bias_statis = np.vstack((bias_statis, tmp))
        self.auto_middle_save(bias_statis, target)

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
    parser.add_argument('--stride', type=int, default=1,
                        help='stride for matching (default=1)')
    parser.add_argument('--velocity_fold', type=str)
    parser.add_argument('--sta_name', type=str)
    parser.add_argument('--target_output', type=str)
    parser.add_argument('--xrange', type=int, nargs=2)
    parser.add_argument('--yrange', type=int, nargs=2)
    parser.add_argument('--GRAtype', type=str, default='np', choices=['np', 'nye','sobel'])
    return parser.parse_args()

def main():
    args = parse_args()
    if args.config:
        cfg = parse_txt_config(args.config)
    else:
        cfg = vars(args)

    def get(k, default=None):
        return cfg.get(k, default)

    stride = int(get('stride'))
    sta_name = str(get('sta_name'))
    velocity_fold = str(get('velocity_fold'))
    target_output = str(get('target_output'))
    jrange = list(map(int, get('xrange')))
    irange = list(map(int, get('yrange')))
    gratype = str(get('GRAtype'))

    start = Line_gradient(velocity_fold, sta_name)
    start.process_line(target_output, irange, jrange, GRAtype=gratype, stride=stride,
                       nyeparas=[1,True,10,nye_maxT,nye_tol,nye_dt0,1/2])

# ------------------------------------------------------------------
if __name__ == '__main__':
    main()
