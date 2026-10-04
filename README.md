
pip install -r requirements.txt

#run these commands

python run_overfit.py --dataset synthetic --epochs 5          # 1)Just fors test. No download here.
python run_overfit.py --dataset cifar_catdog --epochs 60 --stop-acc 0.999   # small binary version first
python run_overfit.py --dataset cifar10 --depth 2 --children 2 --hidden 256 --epochs 100 --stop-acc 0.999
python run_baseline.py --dataset cifar10 --depth 2 --children 2 --hidden 256 --epochs 100
python run_double_descent.py --dataset cifar10 --train-subset 10000 --label-noise 0.15 --epochs 100 --vary hidden
python run_receptive_fields.py --ckpt results/overfit/model.pt
python run_minimize.py --mode sweep --dataset cifar10 --epochs 100
python run_minimize.py --mode prune --ckpt results/overfit/model.pt
```
Cats vs Dogs: use `--dataset cifar_catdog` (CIFAR cat/dog subset) or `--dataset catsdogs`
with images in `data/PetImages/{Cat,Dog}/`. Add `--trunk separate` for one CNN per node.
Everything writes JSON + PNGs under `results/`.
