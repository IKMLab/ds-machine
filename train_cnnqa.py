from DSMachine.dataset.data_helper import DataTransformer
from DSMachine.models.CNNQA import ConvolutionDiscriminator
from DSMachine.trainer.qa_trainer import QATrainer


def main():
    # TODO store all configs in a file
    print("Loading the dataset...")
    data_transformer = DataTransformer(path='dataset/Gossiping/Gossiping-QA-Dataset.txt', min_length=5)
    qa_discriminator = ConvolutionDiscriminator(vocab_size=data_transformer.vocab_size,
                                                embedding_size=300, kernel_sizes=(1, 2, 3,), kernel_num=72,
                                                hidden_size=500, out_size=2, conv_over_qa=False, residual=False).cuda()
    trainer = QATrainer(data_transformer, model=qa_discriminator, checkpoint_path='checkpoint/CNNQA_WITH_RESIDUAL_POS_50_CLEAN_STOPWORDs.pt')
    print("Training the CNN-QA-Discriminator")
    trainer.load_pretrained_model(checkpoint_path='checkpoint/CNNQA_WITH_RESIDUAL_POS_50_CLEAN_STOPWORDs.pt')
    trainer.train(epochs=100)

if __name__ == "__main__":
    main()
