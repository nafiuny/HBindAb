import logging
import pytorch_lightning as pl
from data.dataset import create_dataloader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from transformers import BertTokenizer, AutoModel
import os
from model_lightning import HBindAbLight
import argparse
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
log = logger


def main(args):

    
    pl.seed_everything(42)

    dataloaders = create_dataloader(
        dataset_name=args.dataset,
        ab_tokenizer_name=args.antibody_model_name,
        ag_tokenizer_name=args.antigen_model_name,
        cache=args.cache,
        bsize=args.bsize,
        bsize_eval=args.bsize_eval,
        num_workers=args.num_data_workers
    )

    tb_dir = os.path.join(args.root_dir, args.exp_dir, "tb_logs")
    tb_logger = TensorBoardLogger(tb_dir, version=0)

    checkpoint_dir = os.path.join(args.root_dir, args.exp_dir, 'checkpoints_h3')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(args.checkpoint):
        restart_model = args.checkpoint
    else:
        if args.resume:
            file = os.path.join(checkpoint_dir, 'last.ckpt')
            if os.path.exists(file):
                restart_model = file
            else:
                log.info(f'The model {file} does not exist' )
                restart_model = None
        else:
            restart_model = None
    if args.run == 'train':
        model = HBindAbLight(
            antibody_model_name=args.antibody_model_name,
            antigen_model_name=args.antigen_model_name,
            cache=args.cache,
            lr=args.lr,
            num_samples=args.num_samples
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            every_n_epochs=1
        )
        trainer = pl.Trainer(
            max_epochs=args.max_epochs if hasattr(args, 'max_epochs') else 5,
            accelerator='gpu',
            devices=1,
            default_root_dir=os.path.join(args.root_dir, args.exp_dir),
            logger=tb_logger,
            callbacks=[checkpoint_callback, TQDMProgressBar()]
        )
        trainer.fit(model=model,
                    train_dataloaders=dataloaders['train'],
                    val_dataloaders=dataloaders['validation'],
                    ckpt_path=restart_model)

    elif args.run == 'test':
        if restart_model is None:
            log.info(f'The model checkpoint was not found, cannot do testing')
            return

        model = HBindAbLight.load_from_checkpoint(
            restart_model,
            strict=False
        )

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            default_root_dir=os.path.join(args.root_dir, args.exp_dir),
            logger=tb_logger
        )

        trainer.test(model, dataloaders=dataloaders['test'])
        
    elif args.run == 'generate':
        if restart_model is None:
            log.info(f'The model checkpoint was not found, cannot do inference')
            return


        model = HBindAbLight.load_from_checkpoint(restart_model, strict=False)
        model.eval()
        model = model.to("cuda")

        antibody_seq = args.single_input
        antigen_seq = args.antigen_seq

        fasta_path = "output/model_HbindAb/generate.fasta"

        model.generate_fasta(antibody_seq=antibody_seq, antigen_seq=antigen_seq, fasta_path=fasta_path, num_samples=args.num_samples)
                
        print("FASTA saved to", fasta_path)
    
    else:
        raise ValueError("Invalid run mode. Allowed modes are 'train', 'test', and 'generate'.")
if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument("--run", type=str, default='train')
    parser.add_argument("--loging_level", choices=["debug", "info"], default="info", help="logging level")
    parser.add_argument("--single_input", type=str, default='EVQLQQSGTVLARPGASVKMSCKASGYTFTSYWMHWIKQRPGQGLEWIGAIYPGDSDTKYNQKFKGKAKLTAVTSTSTAYMELSSLTNEDSAVYYC*************WGQGTTLTVSS')
    parser.add_argument("--antigen_seq", type=str, default='LPLLCTLNKSHLYIKGGNASFQISFDDIAVLLPQYDVIIQHPADMSWCSKSDDQIWLSQWFMNAVGHDWHLDPPFLCRNRTKTEGFIFQVNTSKTGVNENYAKKFKTGMHHLYREYPDSCLNGKLCLMKAQPTSWPLQCPLD')
    parser.add_argument("--antibody_model_name", type=str, default='Rostlab/prot_bert')
    parser.add_argument("--antigen_model_name", type=str, default='facebook/esm2_t6_8M_UR50D')
    parser.add_argument('--checkpoint', type=str, default='output/model_HbindAb/checkpoints_h3/last.ckpt')
    parser.add_argument('--root_dir', type=str, default='output')
    parser.add_argument('--exp_dir', type=str, default='model_HbindAb')
    parser.add_argument('--dataset', type=str, default='sabdab')
    parser.add_argument('--bsize', type=int, default=32)
    parser.add_argument('--bsize_eval', type=int, default=32)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--cache', type=str, default='cache')
    parser.add_argument('--num_data_workers', type=int, default=0)
    parser.add_argument('--bar', default=10, type=int)
    parser.add_argument('--max_epochs', type=int, default=70, help='Maximum number of training epochs')
    args = parser.parse_args()

    main(args)
    
