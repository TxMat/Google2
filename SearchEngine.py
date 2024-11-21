from pandas import DataFrame


class SearchEngine:

    def __init__(self, corpus):
        self.term_freq_matrix = corpus.get_tf_matrix()
        self.vocab = corpus.get_vocab()
        self.TFxIDF_matrix = None
        self.corpus = corpus



    # Afin de realiser votre moteur de recherche, les principales etapes sont les suivantes :
    # • demander a l’utilisateur d’entrer quelques mots-clefs,
    # • transformer ces mots-clefs sous la forme d’un vecteur sur le vocabulaire precedemment construit,
    # • calculer une similarite entre votre vecteur requete et tous les documents,
    # 1
    # • trier les scores resultats et afficher les meilleurs resultats.
    # La similarite peut etre calculee `a l’aide d’un simple produit scalaire entre le vecteur requete et le
    # vecteur du texte vise. Une mesure qui est souvent plus appropriee est celle du cosinus (cf. https:
    # //fr.wikipedia.org/wiki/Similarite_cosinus).

    # Pour finir, vous devez integrer tout le code produit dans ce TD dans une classe intitulee SearchEngine.
    # Cette classe doit respecter certaines contraintes :
    # • On doit pouvoir donner un objet de type Corpus lorsqu’on instancie le moteur. La construction
    # de la matrice Documents x Termes doit se faire dans la foulee.
    # • La classe doit proposer une fonction search avec deux arguments : les mots clefs de la requete
    # et le nombre de documents a retourner a l’utilisateur.
    # • Le resultat de la recherche doit etre retournee sous la format d’une table DataFrame de pandas.

    def get_vector(self, query):
        # transform query into vector
        cleaned_query = self.corpus.clean_text(query)
        query_vector = [0] * len(self.vocab)
        for term in cleaned_query.split():
            if term in self.vocab:
                query_vector[self.vocab.index(term)] += 1
        return query_vector

    def search(self, query):
        # transform query into vector
        query_vector = self.get_vector(query)
        # calculate similarity between query vector and all documents
        similarity = self.term_freq_matrix.dot(query_vector)
        # sort results and display the best results
        results = []
        for i, score in enumerate(similarity):
            if score > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.get_data(), score, doc.title, doc.author.name, doc.date, doc.url, doc.body])
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Document", "Score", "Title", "Author", "Date", "URL", "Body"])