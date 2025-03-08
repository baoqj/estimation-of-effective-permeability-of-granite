# 花岗岩渗透率研究的PINN实现 - 完整代码

- eﬀective_permeability_pinn.py 使用方法

    1. 初次训练模型
        这将训练模型，计算渗透率场，并可视化结果。模型状态和渗透率场将保存到磁盘。

        ```bash
        python effective_permeability_pinn.py --train --compute --visualize
        ```

    2. 加载已训练模型并计算渗透率场
        这将加载已训练的模型，计算渗透率场并保存。

        ```bash
        python effective_permeability_pinn.py --compute
        ```

    3. 只可视化之前计算的结果
        这将从保存的数据直接可视化结果，无需重新训练或计算。

        ```bash
        python effective_permeability_pinn.py --visualize
        ```


    4. 仅处理特定岩石
        这将只处理Stanstead花岗岩的数据。

        ```bash
        python effective_permeability_pinn.py --visualize --rock Stanstead
        ```


- eﬀective_permeability_pinn_auto.py 使用方法

    1. 最简单的使用:
        ```bash
        python effective_permeability_pinn.py
        ```
        执行所有操作：训练、计算和可视化，对两种岩石都进行处理。如果找到之前保存的模型，会从保存点继续训练。

    2. 仅训练模型:
        ```bash
        python effective_permeability_pinn.py --train
        ```


    3. 加载已训练模型并计算渗透率场:
        ```bash
        python effective_permeability_pinn.py --compute
        ```


    4. 只可视化训练结果:
        ```bash
        python effective_permeability_pinn.py --visualize
        ```


    5. 指定岩石类型:
        ```bash
        python effective_permeability_pinn.py --rock Stanstead
        ```
        这个修改后的代码通过检查点机制实现了自动从上次终止点继续训练的功能，避免了因训练中断而丢失计算结果的风险。同时，它也能在未发现之前保存的模型时自动从头开始训练，确保整个工作流程的连续性。