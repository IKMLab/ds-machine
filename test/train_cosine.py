from DSMachine.dataset.data_helper import DataTransformer
from DSMachine.models.CNNQA import ConvolutionCosineSimilarity
from DSMachine.trainer.qa_trainer import SimTrainer


def main():
    data_transformer = DataTransformer(path='dataset/Gossiping-QA-Dataset.txt', min_length=6)
    conv_cosine_sim = ConvolutionCosineSimilarity(data_transformer.vocab_size, 256, kernel_sizes=(2,3,4), kernel_num=64, with_linear=False).cuda()
    trainer = SimTrainer(data_transformer, model=conv_cosine_sim, checkpoint_path='checkpoint/CNNQA_WITH_COSINE_DIS_POS_55.pt')
    print("Training the CNN-QA-Cosine-Distance")
    trainer.train(epochs=77)

if __name__ == "__main__":
    main()