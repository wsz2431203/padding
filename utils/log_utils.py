def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    """
    记录训练过程中关键指标的值，包括屏幕输出和 TensorBoard 日志记录。
    """

    # 计算平均损失
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # 打印日志到屏幕
    print(f'epoch: {epoch}, train_batch_id: {batch_id}, avg_cost: {avg_cost}')
    print(f'grad_norm: {grad_norms[0]}, clipped: {grad_norms_clipped[0]}')

    # 检查是否启用 TensorBoard，并确保 tb_logger 有效
    if not opts.no_tensorboard and tb_logger is not None:
        # 记录平均损失
        tb_logger.log_value('avg_cost', avg_cost, step)

        # 记录强化学习损失和负对数似然值
        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        # 记录梯度信息
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        # 如果基线是 'critic'，记录基线相关的损失和梯度
        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
    else:
        # 如果未启用 TensorBoard 或 tb_logger 无效，则打印警告信息
        if tb_logger is None:
            print("Warning: tb_logger is None. Skipping TensorBoard logging.")
