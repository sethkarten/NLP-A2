import matplotlib.pyplot as plt
import numpy as np

def main():
    dim = []
    lr = []
    perp0 = []
    perp1 = []
    with open("nn_data0.txt", 'r') as f:
        for line in f:
            line=line.strip('\n')
            _dim, _lr, _perp = line.split(',')
            dim.append(np.log(int(_dim)))
            lr.append(np.log(float(_lr)))
            perp0.append(np.log(float(_perp)))
    with open("nn_data1.txt", 'r') as f:
        for line in f:
            line=line.strip('\n')
            _dim, _lr, _perp = line.split(',')
            perp1.append(np.log(float(_perp)))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.scatter(lr[0::5], perp0[0::5], color='b', label='1e-05')
    ax1.scatter(lr[1::5], perp0[1::5], color='g', label='3e-05')
    ax1.scatter(lr[2::5], perp0[2::5], color='r', label='0.0001')
    ax1.scatter(lr[3::5], perp0[3::5], color='c', label='0.0003')
    ax1.scatter(lr[4::5], perp0[4::5], color='m', label='0.001')
    ax1.legend()
    ax1.set_title('lr')
    ax1.set_ylabel('Log Perplexity')
    ax1.set_xlabel('Log Learning Rate')
    # ax1.set_xlim(float('1e-06'),0.00101)

    ax2.scatter(dim[:5], perp0[:5], color='b', label='1', marker='+')
    ax2.scatter(dim[5:10], perp0[5:10], color='g', label='5', marker='+')
    ax2.scatter(dim[10:15], perp0[10:15], color='r', label='10', marker='+')
    ax2.scatter(dim[15:20], perp0[15:20], color='c', label='100', marker='+')
    ax2.scatter(dim[20:25], perp0[20:25], color='m', label='200', marker='+')
    ax2.legend()
    ax2.set_title('dim')
    ax2.set_ylabel('Log Perplexity')
    ax2.set_xlabel('Log Dimension')

    fig.suptitle("Linear")
    plt.tight_layout()
    fig.savefig("linear")
    # ax2.set_xlim(1,200)\

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.scatter(lr[0::5], perp1[0::5], color='b', label='1e-05')
    ax1.scatter(lr[1::5], perp1[1::5], color='g', label='3e-05')
    ax1.scatter(lr[2::5], perp1[2::5], color='r', label='0.0001')
    ax1.scatter(lr[3::5], perp1[3::5], color='c', label='0.0003')
    ax1.scatter(lr[4::5], perp1[4::5], color='m', label='0.001')
    ax1.legend()
    ax1.set_title('lr')
    ax1.set_ylabel('Log Perplexity')
    ax1.set_xlabel('Log Learning Rate')
    # ax1.set_xlim(float('1e-06'),0.00101)

    ax2.scatter(dim[:5], perp1[:5], color='b', label='1', marker='+')
    ax2.scatter(dim[5:10], perp1[5:10], color='g', label='5', marker='+')
    ax2.scatter(dim[10:15], perp1[10:15], color='r', label='10', marker='+')
    ax2.scatter(dim[15:20], perp1[15:20], color='c', label='100', marker='+')
    ax2.scatter(dim[20:25], perp1[20:25], color='m', label='200', marker='+')
    ax2.legend()
    ax2.set_title('dim')
    ax2.set_ylabel('Log Perplexity')
    ax2.set_xlabel('Log Dimension')

    fig.suptitle("Non-Linear")
    plt.tight_layout()
    fig.savefig("nonlinear")

    plt.show()
if __name__ == '__main__':
    main()
