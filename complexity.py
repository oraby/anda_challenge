
def complexity(color, trial, bins, path_to_misc):
    """
    Give color as a string, trial as a scalar, bins as a scalar, and path_to_misc as the full path to
    wherever you have SGexercises/misc.py
    Plots: raster plot, spike counts per bin (in ms), the complexity distribution for that bin size,
    complexity distribution for data vs. surrogates, then substracts the two from each other. Finally, three plots
    are created of complexity distributions for data, surrogates, and their difference, scanning across bin sizes
    from 0 to 30 ms.
    """
    import sys
    import numpy
    import quantities as pq
    import elephant.spike_train_correlation as corr
    import elephant.spike_train_surrogates as surr
    import elephant.statistics as stats
    import elephant.conversion as conv
    import elephant.unitary_event_analysis as ue
    import elephant.spike_train_generation as stocmod
    import elephant.spade as spade
    sys.path.append(path)
    import misc
    import matplotlib.pyplot as plt
    import neo

    data_block = ut.load_dataset(color, path=None)
    train = data_block.segments[trial].spiketrains # segments is trials

    binsize = bins
    pophist = stats.time_histogram(train, binsize, binary=True)
    complexity_cpp = stats.complexity_pdf(train, binsize)
    # Plot the results
    fig = plt.figure(figsize=(12,5))
    fig.subplots_adjust(top=.92, bottom=.05, hspace=.5, wspace=.2)
    misc.add_raster(fig, 2, 2, 1, train, ms=1, xlabel='time', ylabel='neuron id')
    misc.add_hist(fig, 2, 2, 3, pophist.times, pophist, pophist.sampling_period, xlabel='time', ylabel='counts')
    ylim = plt.gca().get_ylim()
    plt.subplot(1,2,2)
    # plt.bar(complexity_cpp.times, list(A) + [0]*(n-assembly_size), color='r', label='amplitude distrib.')
    #plt.bar(range(len(A)), A, color='r', label='amplitude distrib.')
    plt.plot(complexity_cpp.times, complexity_cpp, label='complexity distrib.')
    plt.ylim([0, 0.25])
    plt.xlabel('complexity', size=12)
    plt.ylabel('probability', size=12)
    plt.suptitle('train', size=14)
    plt.legend()

    # generation of surrogates
    surr_sts = []

    for st in train:
    surr_sts.append(surr.randomise_spikes(st)[0])

    # Computation of the Complexity Distributions
    complexity_surr = stats.complexity_pdf(surr_sts, binsize)
    diff_complexity = complexity_cpp - complexity_surr

    # Plot the difference of the complexity distributions of the correlated and independent CPP
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(complexity_cpp.times, complexity_cpp, color='blue', label="corr'd")
    plt.plot(complexity_surr.times, complexity_surr, color='red', label="surrogate")
    plt.xlabel('complexity')
    plt.ylabel('probability')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(complexity_cpp.times, diff_complexity)
    plt.xlabel('complexity')
    plt.ylabel('probability diff.')
    plt.show()

    # computation of the complexity distributions of CPP and surrogates for different binsizes
    # and storing the result in matrices
    binsizes = range(1, 31)*pq.ms
    max2 = []
    complexity_cpp_matrix = stats.complexity_pdf(train, binsizes[0]).magnitude
    complexity_surr_matrix = stats.complexity_pdf(surr_sts, binsizes[0]).magnitude
    diff_complexity_matrix = complexity_cpp_matrix - complexity_surr_matrix
    max2.append(numpy.argmax(
    diff_complexity.magnitude[numpy.argmin(diff_complexity.magnitude):]) + numpy.argmin(
                diff_complexity.magnitude))

    for i, h in enumerate(binsizes[1:]):
    complexity_cpp = stats.complexity_pdf(train, h)
    complexity_surr = stats.complexity_pdf(surr_sts, h)
    diff_complexity = complexity_cpp - complexity_surr
    max2.append(numpy.argmax(
        diff_complexity.magnitude[numpy.argmin(
            diff_complexity.magnitude):]) + numpy.argmin(
                diff_complexity.magnitude))

    complexity_cpp_matrix = numpy.hstack(
        (complexity_cpp_matrix, complexity_cpp.magnitude))
    complexity_surr_matrix = numpy.hstack(
        (complexity_surr_matrix, complexity_surr.magnitude))
    diff_complexity_matrix = numpy.hstack(
        (diff_complexity_matrix, diff_complexity.magnitude))

    # Plot the complexity matrices
    inch2cm = 0.3937
    scale = 2  # figure scale; '1' creates a 8x10 cm figure (half a page width)
    label_size = 6 + 1 * scale
    text_size = 8 + 1 * scale
    tick_size = 4 + 1 * scale
    plt.figure(figsize=(8*scale*inch2cm, 12*scale*inch2cm), dpi=100*scale)
    plt.subplots_adjust(right=.97, top=.95, bottom=.2-.08*scale, hspace=.55)

    plt.subplot(3, 1, 1)
    plt.title('CPP complexity')
    plt.pcolor(complexity_cpp_matrix.T)
    plt.colorbar()
    plt.tick_params(length=2, direction='out', pad=0)
    plt.yticks(binsizes[0:-1:3].magnitude)
    plt.ylabel('Binsize')
    plt.xlabel('Complexity')
    plt.xlim([0,complexity_cpp_matrix.T.shape[1]])
    plt.ylim([0,complexity_cpp_matrix.T.shape[0]])
    plt.ylim([binsizes[0], binsizes[-1]])

    plt.subplot(3, 1, 2)
    plt.title('Surrogate complexity')
    plt.pcolor(complexity_surr_matrix.T)
    plt.colorbar()
    plt.ylabel('Binsize')
    plt.xlabel('Complexity')
    plt.tick_params(length=2, direction='out', pad=0)
    plt.yticks(binsizes[0:-1:3].magnitude)
    plt.xlim([0,complexity_cpp_matrix.T.shape[1]])
    plt.ylim([0,complexity_cpp_matrix.T.shape[0]])
    plt.ylim([binsizes[0], binsizes[-1]])

    plt.subplot(3, 1, 3)
    plt.title('Difference of complexities')
    plt.pcolor(diff_complexity_matrix.T)
    plt.colorbar()
    plt.plot(max2, binsizes, 'm')
    plt.ylabel('Binsize')
    plt.xlabel('Complexity')
    #plt.yticks(binsizes[0:-1:3].magnitude)
    plt.tick_params(length=2, direction='out', pad=0)
    plt.xlim([0,complexity_cpp_matrix.T.shape[1]])
    plt.ylim([0,complexity_cpp_matrix.T.shape[0]])
    plt.ylim([binsizes[0], binsizes[-1]])

