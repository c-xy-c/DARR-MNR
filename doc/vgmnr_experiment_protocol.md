# VG-MNR ʵ��Э���������֤����

���ڣ�2026-07-30

���ĵ����� VG-MNR ��ǰ�׶ε���ʽ��֤�����������ж��������ݼ��Ƿ��������� benchmark Ŀ�꣬�����ǽ����������ɡ��ܱ��桢����ͨ����

���ش��������⣺

1. ���ݼ������Ƿ��Ѿ��߱�����һ�ֻ�����֤��������
2. Ӧ��ѡ��Щ�����㷨����ʵ�飿
3. ��θ���ʵ�����ж����ݼ��Ƿ�������Ƿ���Ҫ������

---

## 1. ��ǰ�汾�Ƿ��ʺ�����һ����֤��

���ۣ�**�ʺ�����һ����֤������Ӧ��Ϊ prototype validation set������������ benchmark release��**

��ǰ `mnr_dataset.vgmnr_main` �Ѿ������һ����֤��������������

- ����ģʽ�����ȶ����ɣ�`full`��`no_visual`��`no_context`��`collision`
- `accepted / rejected` ͳ���ѿ�׷��
- `candidate_diagnostics` �����ڼ���ѡ������
- `split_buckets` ���ܰ�����·�ɵ���Ӧ split
- `generation_report.json` �ɸ����������
- `collision` �����ȶ��� `operand_swap` ����

����ζ�ţ�

- ���Զ���һ��С��ģ��֤�����ݼ�
- ���Կ�ʼ�ܻ����㷨
- ���ԱȽϲ�ͬ�㷨�ڲ�ͬ split �ϵı����Ƿ�������Ԥ��

��ǰ�Բ�����ĵط���

- �����ӵ� collision family ������չ��
- ��û����ʵģ�͵�ϵͳ�Ի��߽��
- ��û��֤���� benchmark �Բ�ͬģ�;����ȶ����ֶ�

---

## 2. ʵ��Ŀ��

��ʵ�鲻��Ϊ�����ۡ��ĸ�ģ����á�������Ϊ����֤����������

- `full` �£�ģ��Ӧ�������������Ӿ�����������Ϣ�������
- `no_visual` �£�ģ��Ӧ����������Ϊ�Ӿ���ϵ������
- `no_context` �£�ģ��Ӧ����������Ϊ��������Ϣ������
- `collision` �£�ģ��Ӧ�� `full` �����׻�������Ϊ����ר�Ź���ĳ�ͻ��
- �� heuristic ��Ӧ���׽ӽ���ʵģ��
- ��� baseline Ӧ�������ˮƽ

�����ЩԤ�ڲ�������˵�����ݼ�����Ҫ������������ֱ�ӽ����������¡�

---

## 3. ����ʵ��� 5 �������㷨

������ 5 ���㷨�Ǳ��ֽ������ʵ��ľ��� baseline�����ǲ��ǳ�����𣬶��ǿ���ʵ��ʵ�֡�ʵ���ܵ��㷨��

### 3.1 Random baseline

**����**�������պ�ѡ������Ϣ�����������ġ����� query��

**���**�����ѡ�� 8 ѡ 1 �Ĵ𰸡�

**����**��

- ��Ϊ��������
- ������ݼ��Ƿ�������Ա�ǩй¶

**Ԥ��**��

- ׼ȷ�ʽӽ� 12.5%��8-way �����

---

### 3.2 Candidate-only heuristic

**����**��ֻ�� `answer_set_images`�����������ĺ� query��

**ʵ�ֽ���**��

- �Ժ�ѡͼ����ͼ��ͳ�ƻ�̶�λ��������ȡ
- ʹ����򵥵Ĵ�ֹ��������������

**����**��

- ����ѡ���֡���ɫ���߿�λ�õ��Ƿ�й¶��
- �жϵ�ǰ��ѡ�����Ƿ����û�� shortcut

**Ԥ��**��

- Ӧ���Ը������������Ӧ�ӽ���ʵģ��
- ��������ߣ�˵����ѡ��ƴ���ƫ��

---

### 3.3 Number-only heuristic / numeric shortcut baseline

**����**��ֻʹ�ÿɴ������г�ȡ������ֵ�����ṹ��Ϣ�������������Ӿ���ϵ��

**ʵ�ֽ���**��

- �� metadata ����ȡ��ʽ��ֵ����
- ���ͼ�����ü� OCR / ��ֵ��ȡ���ƻָ���ֵ����������²�

**����**��

- ��������Ƿ��ܱ���ֻ�����֡������ṹ�����
- ��֤�Ӿ����Ƿ���ı�Ҫ

**Ԥ��**��

- �� `full` ������һ��Ч��������Ӧ�ӽ�����ģ��
- �� `no_visual` / `collision` ��Ӧ�����½�

---

### 3.4 Small CNN baseline

**����**������ͼ�񣬶˵����� 8-way ���ࡣ

**ʵ�ֽ���**��

- һ����������������
- �� context/query/candidate ͼ�����������ں�

**����**��

- �ṩһ��������ͼ��ѧϰ baseline
- ��������Ƿ��ܱ��ֲ�������ͼ��������

**Ԥ��**��

- Ӧ���� random �ͼ� heuristic
- ��ͨ�����ڸ�ǿ���Ӿ� transformer / ��ģ̬ģ��

---

### 3.5 ViT-style or multimodal baseline

**����**������������ʹ�ø�ǿ���Ӿ����������ģ̬����ģ�͡�

**ʵ�ֽ���**��

- ViT image encoder
- ���ֳ� VLM / multimodal transformer

**����**��

- ��Ϊ��ǿ baseline����֤ benchmark �Ƿ������ָ��߲������
- ��� `full` / `no_visual` / `no_context` / `collision` �Ƿ���ֺ����Ѷ��ݶ�

**Ԥ��**��

- �� `full` �����
- �� `no_visual`��`no_context`��`collision` ���½�
- ����½������ԣ�˵�� split ��ƻ�������Ч

---

## 4. Ϊʲôֻѡ�� 5 ��

�� 5 �� baseline ����������ؼ����գ�

1. **���޶���**��Random baseline
2. **shortcut ����**��Candidate-only / Number-only
3. **ѧϰ��������**��Small CNN / ViT-style or multimodal baseline

���Ѿ��㹻�жϵ�ǰ benchmark �Ƿ����㡰�����ֶȡ��ܷ�ӳ������Ρ��Ļ���Ҫ��

���һ��ʼ�Ͱ��㷨��������̫�󣬷������õ��Գɱ�������������ж����ݼ����⡣

---

## 5. ʵ������

### 5.1 ���ݰ汾

����̶�һ����֤�����ݼ������ٱ��ܱ߸Ĺ���

���Ҫ��

- `full`
- `no_visual`
- `no_context`
- `collision`

��ʹ��ͬһ�� schema �ͱ����ʽ��

### 5.2 ���ⷽʽ

ÿ���㷨��Ҫ��ÿ�� split ����׼ȷ�ʣ�

- `full_supervision`
- `visual_ablation`
- `context_ablation`
- `pair_collision`

�������¼��

- overall accuracy
- per-split accuracy
- candidate-only accuracy
- error breakdown

### 5.3 ͳ�ƿھ�

����ʹ�ã�

- sample-level accuracy
- pair-level consistency for collision
- rejection statistics for data generation

�������������������������� calibration��confidence gap��oracles ��ָ�ꡣ

---

## 6. ʲô���Ľ��������ģ�

���������ǡ�����ģ�Ͷ��߷֡������ǡ����������Թ��ܵ�Ԥ�ڡ���

### 6.1 Ԥ������

һ�㽨��۲��������Թ�ϵ��

```text
Random < Candidate-only / Number-only < Small CNN < ViT-style / multimodal
```

�������ݶȲ�������˵�����ݼ�������������㡣

### 6.2 �� split ����������

#### `full_supervision`
- Ӧ�������׵� split ֮һ
- ǿģ������������ģ��

#### `visual_ablation`
- ��� `full_supervision` Ӧ�½�
- ���û�½���˵���Ӿ���ϵ�����ò���

#### `context_ablation`
- ��� `full_supervision` ҲӦ�½�
- ���û�½���˵����������Ϣ�������ؼ�

#### `pair_collision`
- Ӧ�ñ� `full_supervision` ���ѣ����ٲ�����ȫ�޲��
- ���� `full_supervision` ��ƽ��˵�� collision ��������ս

### 6.3 heuristic ���������

- Candidate-only / Number-only Ӧ�����Ժ������
- ����Ӧ�ӽ���ʵģ��
- ��� heuristic ��ǿ��˵������ shortcut

---

## 7. ʲô���Ľ���ǲ������ģ�

### 7.1 ����� heuristic ���ܸ�

˵����

- �������������ƫ��
- ��ѡ�������й¶��ǩ
- ���ݼ����� shortcut

��������

- �ĺ�ѡ����
- ����ѡλ��/��ɫ/�ߴ�й¶
- ���͹̶�ģʽƫ��

### 7.2 �ĸ� split ���ּ���һ��

˵����

- ablation û�������ı���Ϣ�ṹ
- split ֻ�����ֲ�ͬ��ʵ��û����

��������

- ǿ���Ӿ���Ҫ��
- ǿ�������ı�Ҫ��
- �� collision ������������Ѷ�

### 7.3 ����ģ�Ͷ��ӽ����

˵����

- ����̫�ѻ���Ϣ����
- �������ɹ��ڿ���

��������

- ���������Ӷ�
- ��ǿ�ṹ�ɽ���
- �������������

### 7.4 ǿģ�ͺ���ģ�Ͳ��̫С

˵����

- benchmark ���ֶȲ���
- ����̫���׻� shortcut ��ǿ

��������

- ��ǿ collision ���Ӷ�
- �����ϸ split
- ������ѡ�븺��������

---

## 8. ����

��ǰ VG-MNR ���ݼ�����**�Ѿ�����֧�ֵ�һ����ʽ��֤**�������������ʵ��˳��

1. �̶���ǰ�ȶ����ݰ汾
2. �� Random baseline
3. �� Candidate-only / Number-only heuristic
4. �� Small CNN
5. �� ViT-style / multimodal baseline
6. �����ĸ� split �������Ƿ�Ԥ���½�

---

## 9. ����ʵ��״̬���ܽ�

�������ļ��ʾ�����ɼ���չ�� 500+ ���ģ�Ա��������������ϣ��� benchmark ���صĶԱ��ԡ�

### 9.1 ���浽�Ĺؼ����

- **shortcut �Ѿ����̬**��Candidate-only �� Number-only ���� random �ӽ���
- **collision �Ѿ�������**��A/B ����ͬһ���Ӿ��ṹ��latent program ���
- **no_visual �Ѿ�ȷ��ս���**��leaf ����Ҳ��ն�
- **Small CNN ��Ӧ���һ���΢�ź�**��full/split ���뷨���Ϊ���
- **Tiny ViT �ձ�û��Ӧ������Ӱ��**��ͨ���뿼�� more data / better architecture / better training

### 9.2 500+ ��ģ�޷���

- `full_supervision`: 500
- `visual_ablation`: 500
- `context_ablation`: 500
- `pair_collision`: 1000

### 9.3 ���Ͻ���

- Random: 0.110~0.123
- Candidate-only: 0.114~0.136
- Number-only: 0.118~0.130
- Small CNN: **0.133~0.162**
- Tiny ViT: 0.108~0.128

### 9.4 �ٷ���������

- ��� benchmark ��Ϊ�ı����� shortcut leakage
- ��� split ���ӶȲ��� `pair_collision` �� `full_supervision` ����
- ��� ViT style �޷������ӻ���Ҫ�����߶�
- ��� Small CNN ���� stable ����������Ӧ�ܾ��ܲ�������

### 9.5 ���ڽ���

- ������ benchmark ���Ѿ���չ�����г�����
- �������ӿ���ץ�ļ������ܶȣ��� and future stronger models
- �������继续增强 split 设计，或转向更大的模型/更多训练数据

---

## 10. ����

��ǰ VG-MNR ����������**�����ڻ��� prototype validation set**��
���Ѿ�������ģ�ʵ�����ơ���������ḻ��� collision 设计��
��Ҫ���Ϊ production benchmark，��Ҫ��ӳ�ֶ�和强模型区分度方面进一步加强。
