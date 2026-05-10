from segmenter import CellSegmenter


if __name__ == '__main__':
    c = CellSegmenter('composite_test_0000.png')
    c.segment()
    c.Mix()

    c.Metrics()

    c.clearAndCopy()