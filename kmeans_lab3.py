
import argparse
import csv
import math
import random
from pathlib import Path


def euclid2(a, b):
    return sum((x-y)**2 for x,y in zip(a,b))


def initial_centers_kmeanspp(points, k, seed):
    rng=random.Random(seed)
    centers=[rng.choice(points)]
    while len(centers)<k:
        weights=[min(euclid2(p,c) for c in centers) for p in points]
        total=sum(weights)
        if total == 0:
            candidates=[p for p in points if p not in centers]
            centers.append(rng.choice(candidates or points)); continue
        draw=rng.random()*total; cumul=0
        choice=points[-1]
        for p,w in zip(points,weights):
            cumul+=w
            if cumul>=draw:
                choice=p; break
        centers.append(choice)
    return [tuple(c) for c in centers]


def kmeans(points, k=3, seed=42, tol=1e-9, max_iter=100):
    if not points or k<1 or k>len(points):
        raise ValueError('Некорректное число кластеров или пустые данные')
    points=[tuple(map(float,p)) for p in points]
    if len(set(map(len,points)))!=1 or len(points[0])<1:
        raise ValueError('Все точки должны иметь одинаковую ненулевую размерность')
    centers=initial_centers_kmeanspp(points,k,seed)
    labels=[-1]*len(points)
    history=[]
    for iteration in range(max_iter):
        new_labels=[min(range(k), key=lambda i: euclid2(p,centers[i])) for p in points]
        new_centers=[]
        for i in range(k):
            members=[p for p,l in zip(points,new_labels) if l==i]
            if members:
                new_centers.append(tuple(sum(p[d] for p in members)/len(members) for d in range(len(points[0]))))
            else:
                # If empty, reposition to the point with largest current reconstruction error.
                furthest=max(points,key=lambda p:min(euclid2(p,c) for c in centers))
                new_centers.append(furthest)
        inertia=sum(min(euclid2(p,c) for c in new_centers) for p in points)
        history.append(inertia)
        movement=max(math.sqrt(euclid2(c,n)) for c,n in zip(centers,new_centers))
        labels=new_labels; centers=new_centers
        if movement < tol:
            break
    # Ensure assignments correspond to final centers.
    labels=[min(range(k), key=lambda i: euclid2(p,centers[i])) for p in points]
    final_inertia=sum(euclid2(p,centers[lab]) for p,lab in zip(points,labels))
    if abs(history[-1]-final_inertia)>1e-8:
        history.append(final_inertia)
    return centers, labels, history


def example_points(seed=2026, count_per_cluster=40):
    rng=random.Random(seed)
    true_centers=[(-3,0),(0,3),(3,0)]
    points=[]; true_labels=[]
    for idx,(x,y) in enumerate(true_centers):
        for _ in range(count_per_cluster):
            points.append((rng.gauss(x,1.0),rng.gauss(y,1.0)))
            true_labels.append(idx)
    return points,true_labels


def save_plots(points, labels, centers, history, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5))
    for i in range(len(centers)):
        group=[p for p,l in zip(points,labels) if l==i]
        if group:
            ax.scatter([p[0] for p in group],[p[1] for p in group],s=19,label=f'Кластер {i+1}',alpha=.85)
    ax.scatter([p[0] for p in centers],[p[1] for p in centers],marker='X',s=175,c='black',label='Центры')
    ax.set(title='Кластеризация методом k-средних',xlabel='Признак X',ylabel='Признак Y')
    ax.legend(loc='best'); ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(directory/'Кластеры.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(range(1,len(history)+1),history,marker='o')
    ax.set(title='Сходимость k-means',xlabel='Итерация',ylabel='Сумма квадратов расстояний (inertia)')
    ax.grid(alpha=.25); ax.set_xticks(range(1,len(history)+1))
    fig.tight_layout();fig.savefig(directory/'Сходимость.png',dpi=170);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='ЛР №3: метод k-средних')
    ap.add_argument('--k',type=int,default=3)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'output')
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    points, original=example_points()
    centers,labels,history=kmeans(points,k=args.k)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'Данные_и_кластеры.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['x','y','исходная_группа','кластер_kmeans'])
        for p,a,b in zip(points,original,labels): writer.writerow([f'{p[0]:.5f}',f'{p[1]:.5f}',a,b])
    print(f'Точек: {len(points)}; k={args.k}; итераций: {len(history)}')
    for i,c in enumerate(centers):
        print(f'Центр {i+1}: ({c[0]:.4f}; {c[1]:.4f}), точек: {labels.count(i)}')
    print('Сходимость:', ' -> '.join(f'{x:.4f}' for x in history))
    print(f'Итоговая инерция: {history[-1]:.4f}')
    if not args.no_plots:
        save_plots(points,labels,centers,history,args.output)
        print('Графики сохранены в',args.output)

if __name__ == '__main__':
    main()
