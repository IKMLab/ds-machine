from DSMachine.dataset.data_helper import SentimentDataTransformer
from DSMachine.sentiment.sentiment_classify import GreedyHAN
from DSMachine.trainer.supervised_trainer import GreedyHANTrainer

def main():
    data_transformer = SentimentDataTransformer(dataset_path='dataset/Weibo/weibo_with_sentiment_tags.data',
                                                emote_meta_path='dataset/Weibo/emote/sentiment_class.txt',
                                                use_cuda=True,
                                                char_based=True)

    classifier = GreedyHAN(vocab_size=data_transformer.vocab_size, embedding_size=256,
                                        hidden_size=256, output_size=4,
                                        kernel_sizes=(1,2,3), kernel_num=64)
    classifier = classifier.cuda()
    trainer = GreedyHANTrainer(data_transformer ,classifier, checkpoint_path='checkpoint/GreedyHanWith3MAX.pt')
    trainer.train(epochs=3000, batch_size=128, pretrained=False)

if __name__ == '__main__':
    main()