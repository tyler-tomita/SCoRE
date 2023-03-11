def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def moving_average(x, window_size):
    x_cs = np.cumsum(x)
    n_cs = len(x_cs)
    x_ave = torch.tensor([x_cs[window_size-1]/window_size] + [(x_cs[i+window_size] - x_cs[i])/window_size for i in range(n_cs-window_size)])
    return(x_ave)

# unpickle cifar binary files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def computeRunningAverage(accuracy_array, window_size):
    ave_acc = torch.zeros(len(accuracy_array))
    if type(accuracy_array) == list:
        accuracy_array = torch.tensor(list)
    for i in range(len(accuracy_array)):
        idx_low = np.max([0, i+1-window_size])
        idx_high = i+1
        ave_acc[i] = accuracy_array[idx_low:idx_high].mean()
    return ave_acc