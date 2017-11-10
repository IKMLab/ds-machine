from DSMachine.dataset.data_helper import SentimentDataTransformer, PTTSentimentDataTransformer
from DSMachine.sentiment.sentiment_classify import RNNSentimentClassifier
from DSMachine.trainer.supervised_trainer import ClassifierTrainer

def main():
    data_transformer = PTTSentimentDataTransformer(dataset_path='dataset/OpenDataShare/',
                                                use_cuda=True,
                                                char_based=False)

    classifier = RNNSentimentClassifier(vocab_size=data_transformer.vocab_size, hidden_size=512, output_size=2, layers=1)
    classifier = classifier.cuda()
    trainer = ClassifierTrainer(data_transformer ,classifier, learning_rate=1e-3)
    #trainer.train(epochs=3000, batch_size=128, pretrained=False, binary=True)
    trainer.cv_train(k=10, epoch=25, batch_size=512, sigmoid=False, cv_checkpoint='checkpoint/rnn_word_cv.pt')


if __name__ == '__main__':
    main()
