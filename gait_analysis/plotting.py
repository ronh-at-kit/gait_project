import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pickle as pl
import os


plt.style.use('seaborn')
fs = 16


def change_fontsize(ax, fs):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fs)

sample_data = {
    'label1' : np.random.randint(2, 10, size=(10)),
    'label2' : np.random.randint(2, 10, size=(10)).astype(np.float)/ 100,
}
sample_data = {key: (val, None) for key, val in sample_data.iteritems()}

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if type(height) in [np.int, np.uint, int, np.int8, np.int16, np.int32, np.int64]:
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom', fontsize=fs)
        else:
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '{:.02f}'.format(height),
                    ha='center', va='bottom', fontsize=fs)


def barplot(fig=None, ax=None, data_dict=sample_data, height_text=True, bar_spacing=1.0):
    '''
    :param fig:
    :param ax:
    :param data_dict: { label1 : (heights1, yerr1), label2: (heights2, yerr2), 'x' : x_data}
    :return:
    '''

    for key, val in data_dict.iteritems():
        assert type(val) == tuple

    x = data_dict.pop('x', None)
    if x is None:
        x = np.arange(len(data_dict.values()[0][0])) * float(bar_spacing)

    num_bars = float(len(data_dict.values()))
    width = 1/(num_bars + 1) * bar_spacing

    i = 0
    for label, data in data_dict.iteritems():
        height, yerr = data
        if yerr is None:
            yerr = np.zeros_like(height)
        r = ax.bar(x + i*width - 0.5 + width/2,
                   yerr = yerr, height=height, width=width,
                   label=label, align='center')
        if height_text:
            autolabel(r, ax)
        i += 1
    ax.legend(fontsize=fs)
    ax.set_xticks(x)
    change_fontsize(ax, fs)
    return fig, ax


class FigSaver():
    def __init__(self, save_dir='.', plt_savefig_format='png', plt_savefig_kwargs={'dpi' : 250}):
        self._save_dir = save_dir
        self._cur_fig_num = 0
        self._plt_savefig_kwargs=plt_savefig_kwargs
        self._plt_savefig_format = plt_savefig_format

    def save_fig(self, fig, fname=''):
        '''

        :param fig:
        :param fname: file name without extension. will automatically get an ordering number beforehand
        :return:
        '''
        fname = '{:02d}_{}'.format(self._cur_fig_num, fname)
        pkl_fname = os.path.join(self._save_dir, '{}.{}'.format(fname, 'pickle'))
        plt_savefig_fname = os.path.join(self._save_dir, '{}.{}'.format(fname, self._plt_savefig_format))
        pl.dump(fig, file(pkl_fname, 'w')) # this might only work for python 2.7
        plt.savefig(plt_savefig_fname, **self._plt_savefig_kwargs)
        return self

    def next(self):
        '''
        increment figure counter. This method returns self so that you can chain methods goteher like
        fig_saver.next().next()
        :return:
        '''
        self._cur_fig_num += 1
        return self

    def set(self, value):
        # TODO magic function?
        self._cur_fig_num = value
        return self

    def reset(self):
        self._cur_fig_num = 0
        return self


if __name__ == '__main__':
    fig, ax = plt.subplots()
    fig, ax = barplot(fig, ax, bar_spacing=5)
    fig.show()

    f_saver = FigSaver()
    f_saver.save_fig(fig, 'test')