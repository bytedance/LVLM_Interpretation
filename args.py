import argparse


def init_args():

    parser = argparse.ArgumentParser(
        description='Generate explanations for open-ended responses of LVLMs.'
    )

    parser.add_argument(
        '--model',
        metavar='M',
        type=str,
        default='llava',
        help='The model to use for making predictions.')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cvbench',
        help='The dataset to use for making predictions.')

    parser.add_argument(
        '--output_dir',
        type=str,
        help='The path to the output directory for saving explanation results.',
        required=True)

    parser.add_argument(
        '--size',
        type=int,
        default=32,
        help='The resolution of mask to be generated.')

    parser.add_argument(
        '--input_size',
        type=int,
        default=336,
        help='The input size to the network.')

    parser.add_argument(
        '--manual_seed',
        type=int,
        default=0,
        help='The manual seed for experiments.')

    parser.add_argument(
        '--method',
        type=str,
        default='iGOS+'
    )

    parser.add_argument(
        '--opt',
        type=str,
        default='NAG',
        help='The optimization algorithm.'
    )

    parser.add_argument(
        '--diverse_k',
        type=int,
        default=1)

    parser.add_argument(
        '--init_posi',
        type=int,
        default=0,
        help='The initialization position, which cell of the K x K grid will be used to initialize the mask with nonzero values (use init_val to control it)')
    """
            If K = 2:      If K = 3:
            -------        ----------
            |0 |1 |        |0 |1 |2 |
            -------        ----------
            |2 |3 |        |3 |4 |5 |
            -------        ----------
                           |6 |7 |8 |
                           ----------
    """

    parser.add_argument(
        '--init_val',
        type=float,
        default=0.,
        help='The initialization value used to initialize the mask in only one cell of the K x K grid.')

    parser.add_argument(
        '--L1',
        type=float,
        default=0.1,
        help='The weight of L1 norm'
    )

    parser.add_argument(
        '--L2',
        type=float,
        default=1.0,
        help='The weight of L2 norm'
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.2,
        help='The exponential decay rate of the graduated non-convexity'
    )

    parser.add_argument(
        '--L3',
        type=float,
        default=20.0,
        help='The weight of BTV norm'
    )
    
    parser.add_argument(
        '--momentum',
        type=int,
        default=3
    )

    parser.add_argument(
        '--ig_iter',
        type=int,
        help='The step size of the integtated gradient accumulation.',
        default=5)

    parser.add_argument(
        '--iterations',
        type=int,
        default=5
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=10,
        help='The step size for updating the mask'
    )

    parser.add_argument(
        '--model_base',
        default=None,
        type=str,
        help='The path to the model base file to be used.'
    )

    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="The path to the input question file."
    )

    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="The path to the image folder"
    )

    # LVLM generation settings
    parser.add_argument("--temperature", type=float,default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--use_yake", type=bool, default=False, help="Whether use yake to detect keywords")
    parser.add_argument("--choices", type=bool, default=False, help="Whether ask the LVLMs to generate single choice instead of open-ended responses")
    parser.add_argument("--ablation_zero", type=bool, default=False, help="Ablation of baseline image using all-zero images")
    parser.add_argument("--ablation_noise", type=bool, default=False, help="Ablation of baseline image using random noise")

    return parser.parse_args()
