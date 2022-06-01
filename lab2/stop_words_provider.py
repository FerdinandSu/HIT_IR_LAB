from model_io import ensure_stop_words


class StopWordsProvider(object):
    def __init__(self):
        self.__p_stop_words = None


    @property
    def _stop_words(self):
        if self.__p_stop_words == None:
            self.__p_stop_words = ensure_stop_words()
        return self.__p_stop_words

