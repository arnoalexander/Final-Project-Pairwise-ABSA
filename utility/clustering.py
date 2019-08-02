"""
Cluster entities
"""

from sklearn.cluster import KMeans
import pickle


class Clusterer:

    """
    Wrapper class for nltk KMeans. General vector clusterer.
    """

    def __init__(self, n_clusters, *args, model_base=None, model_filename=None, **kwargs):
        """
        Initialize object.
        Set model_filename (path to saved base clustering model) OR model_base(initialized clustering model).
        If both are specified, model_base is preferred. If none, new model will be generated based on the remaining
        parameter.
        """
        if model_base is not None:
            self.model = model_base
        elif model_filename is not None:
            self.load(path=model_filename)
        else:
            self.model = KMeans(n_clusters=n_clusters, *args, **kwargs)

    def save(self, path):
        """
        Save base model to a file
        """
        with open(path, 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def load(self, path):
        """
        Load base model from a file
        """
        with open(path, 'rb') as infile:
            self.model = pickle.load(infile)

    def fit(self, X, *args, **kwargs):
        """
        Fit data to clustering model
        """
        self.model.fit(X, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        """
        Get cluster ID of data
        """
        return self.model.predict(X, *args, **kwargs)


if __name__ == '__main__':
    data = [[1, 2], [4, 6], [0, 0], [6, -10], [0, -8.5]]
    clusterer = Clusterer(n_clusters=2)
    clusterer.fit(data)
    print(clusterer.predict(data))
    print(clusterer.model.cluster_centers_)
