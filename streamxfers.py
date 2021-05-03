from functions import read_xfers

class XferIterator():
    def __init__(self, xfersStreamer):
        self.xs = xfersStreamer
        self.events = list(set(self.xs.xfers.submitted).union(set(self.xs.xfers.ended)))
        self.events.sort()
        self.events = self.events[1:]

    def next(self):
        '''Get the next events, two lists with the submitted and ended xfers'''
        try:
            t = self.events.pop(0)
        except IndexError:
            raise StopIteration
        return t, self.xs.get_submitted(t), self.xs.get_ended(t)

    def __next__(self):
        return self.next()


class XferStreamer():
    def __init__(self, xfersfile, nrows=0, skiprows=1):
        self.xfers = read_xfers(xfersfile, nrows=nrows, skiprows=skiprows)

    def get_submitted(self, timestamp):
        return self.xfers.loc[self.xfers.submitted == timestamp]

    def get_ended(self, timestamp):
        return self.xfers.loc[self.xfers.ended == timestamp]

    def __iter__(self):
        return XferIterator(self)

#xs = XferStreamer('data/transfers-FTSBNL-20181001-20181016.csv', 100000)

