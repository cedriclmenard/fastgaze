

def progress_bar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def print_progress_bar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    print_progress_bar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        print_progress_bar(i + 1)
    # Print New Line on Complete
    print()
    """ sample usage:
    import time

    # A List of Items
    items = list(range(0, 57))

    # A Nicer, Single-Call Usage
    for item in progress_bar(items, prefix = 'Progress:', suffix = 'Complete', length = 50):
        # Do stuff...
        time.sleep(0.1)
    """

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", print_every_x_percent=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    should_print = True
    if print_every_x_percent is not None and iteration != total and iteration != 0:
        percent_value = 100.0 * (float(iteration) / float(total))
        previous_percent_value = 100.0 * ((float(iteration - 1)) / float(total))
        if int(percent_value / print_every_x_percent) == int(previous_percent_value / print_every_x_percent):
            should_print = False

    if should_print:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
    """ sample usage:
    import time

    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    print_progress_bar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i, item in enumerate(items):
        # Do stuff...
        time.sleep(0.1)
        # Update Progress Bar
        print_progress_bar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    """

