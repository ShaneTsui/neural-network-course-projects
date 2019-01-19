from utils.display.display import *

class PCA:
    def __init__(self):
        None

    def learn(self, raw):
        # Subtract the mean
        self.mu = raw.mean(axis=0)
        raw -= self.mu

        #         # Calculate covariance
        cov = np.dot(raw, raw.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        self.std = np.sqrt(eig_vals)[1:]

        # Map the eigenvectors to original ones
        eig_vecs = np.dot(raw.T, eig_vecs)[:, 1:]

        # Normalization
        self.transformer = eig_vecs / np.linalg.norm(eig_vecs, 2, axis=0)

    def run(self, data, n_components, normalize=True):
        #         data = data - self.mu
        data = np.dot(data, self.transformer[:, -n_components:])
        if normalize:
            data = data / self.std[-n_components:]
        return data

    def plt_eig_faces(self, n_faces=6, layout=(2, 3)):
        # display eigenfaces
        display_faces(self.transformer.T[0:n_faces], layout=layout,
                      labels=["eigenface {}".format(i) for i in range(n_faces)])
        plt.savefig('Eigenfaces.png', bbox_inches='tight')