#!/usr/bin/python
import argparse
import subprocess
import sys


def main(args):
    if args.inputs is None:
        raise Exception('Please provide input file')
    f = args.inputs[0]

    # Find style sheet file path.
    style = ''
    for _ in f.split('/')[:-1]:
        if _ != '.':
            style += '../'
    style += 'style.css'

    fout = f.replace('.md', '.html')
    with open('temp', 'w') as ftmp:
        if args.pandoc:
            p = subprocess.Popen(
                ['pandoc', f, '-t', 'html'],
                stdout=ftmp,
                stderr=subprocess.PIPE)
        else:
            p = subprocess.Popen(
                ['Markdown.pl', f], stdout=ftmp, stderr=subprocess.PIPE)
        out, err = p.communicate()

    with open('temp', 'r', encoding='utf-8') as ftmp:
        out = ftmp.read()

    with open('template.html', 'r', encoding='utf-8') as tempf:
        s = tempf.read()

    s = s % (style, out)

    with open(fout, 'w', encoding='utf-8') as fout_:
        fout_.write(s)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build markdown')
    parser.add_argument('inputs', type=str, help='input file', nargs=1)
    parser.add_argument(
        '--pandoc',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='whether to use Pandoc')
    args = parser.parse_args()
    main(args)
