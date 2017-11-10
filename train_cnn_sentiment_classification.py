from DSMachine.dataset.data_helper import SentimentDataTransformer, PTTSentimentDataTransformer
from DSMachine.sentiment.sentiment_classify import CNNSentimentClassifier
from DSMachine.trainer.supervised_trainer import ClassifierTrainer

def main():
    # data_transformer = SentimentDataTransformer(dataset_path='dataset/Weibo/weibo_with_sentiment_tags.data',
    #                                             emote_meta_path='dataset/Weibo/emote/sentiment_class.txt',
    #                                             use_cuda=True,
    #                                             char_based=False)

    data_transformer = PTTSentimentDataTransformer(dataset_path='dataset/OpenDataShare/',
                                                use_cuda=True,
                                                char_based=True)

    classifier = CNNSentimentClassifier(vocab_size=data_transformer.vocab_size, embedding_size=512,
                                        hidden_size=768, output_size=2,
                                        kernel_sizes=(1,2,3), kernel_num=128)
    classifier = classifier.cuda()
    trainer = ClassifierTrainer(data_transformer ,classifier, learning_rate=1e-4, checkpoint_path='checkpoint/99acc.pkt')
    #trainer.train(epochs=50000, batch_size=512, pretrained=False, binary=True, sigmoid=False)
    trainer.cv_train(k=10, epoch=15, batch_size=768, sigmoid=False, cv_checkpoint='checkpoint/cv_cnn_char_tt.pt')

if __name__ == '__main__':
    main()
