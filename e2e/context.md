主要是看看vllm支持不支持了

`RFC #42109「Disaggregated Speculative Decoding with Standalone Parallel Draft Model」`，没有相应实现


---

agent告诉我有vllm插件这个东西，这里的plan是

1. 基于vllm，通过插件增加下面的功能
	1. draft：生成gamma个token，kv回滚(prefix cache上即可)
	2. target：verify gamma token，返回结果，kv回滚，这部分用插件覆盖ngram path
2. target 被多个draft共享
	1. 用一个后端服务包target vllm server，请求路由和 `(draft_id, seq_id)` 的应答分发.
    agent补充：请求归target所有


但网络怎么做不知道，从zmq到http，先包装成抽象层，具体咋干再说

### stage 0 

测试vllm插件

这里想验证的是能否修改vllm里的draft，用custom_class(vllm 0.22.0) 注册一个remotedrafter，现在可以replay一些预先录制的draft sequence。

环境基于 8×L40S (sm_89, 48GB) 

### stage 1

实现draft和target的传输层，一个抽象的传输层，底层实现可以换http（慢）或者zmq/grpc（快），先写一个最方便的

现在有一个InProcessDraftClient，是个伪传输层。

对的一半：它不搬字节。没有 socket、没有序列化、没有 IPC，就是一次普通 Python 方法调用。作为「传输」它确实是空操作，所以它测不出序列化开销、排队、TCP 行为——真网络的那些东西一个都没有。

不对的一半：它不是测试替身，它是一个真实的部署模式。

### stage 2

真 draft和target 1 - 1交互

2a fused：真 draft 模型 + InProcessDraftClient（不碰网络）。门禁 = 无损性，fp32 下 γ∈{1,2,4,8} 逐 token 相同。
2b 分布式：同样的模型 + ZMQ。接受率此时已知正确，所以这步只测传输。门禁 = TPOT 随 RTT 线性上升，斜率 ≈ 1/L̄——不线性就说明 propose() 掉出了关键路径（K2 的运行时复查）

### stage 3 

多draft对1target, routing策略，metric收集