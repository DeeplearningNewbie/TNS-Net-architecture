clc
clearvars -except rssi rx_h_est_vec
syms t 
format long
syms sigma

%*****************************************
%初始化顾客源
%*****************************************
%总仿真时间
Total_time = 1;
%队列最大长度
N=10000000000;
t0=0.1*10^(-3);
%TRBA时隙分配策略 每一帧长度为0.5ms
Data = [];
%%0616
D=[1000,1000,1000]; Pow=[70,50,30]*10^(-3); B=0.24*[10^6,10^6,10^6]; N0=4*10^(-21);N_RB=100; %h=raylrnd(\sigma,[m*n])
%任务每位要求的CPU周期数L CPU计算频率Fc
L=[1,1,1]*10^3; Fc=3*10^10;
Tc=D.*L/Fc;
%Fc./(D.*L)
%到达率与服务率
lambda= [2.5,3,3.5]*10^3;



%fun =@(x)B.*log2(1+P.*x./N0./B).*x./sigma.^2.*exp(-x.^2./2*sigma.^2);
%trans=double(int(fun,x,0,inf));
UEs = length(lambda);
mean_mu = sum(D.*L)/UEs;
class1= [];
% for j = 1:length(Tau)
%     tau(j)=sum(Tau(1:j));
% end
arr_num = 0;
for j = 1:UEs
    change_num = round(Total_time*lambda(j)*2);
    if arr_num<change_num
        arr_num = change_num;
    end
end
rssi1 = rx_h_est_vec(:,:,1);
for i = 2:12
    rssi1 = rssi1 + rx_h_est_vec(:,:,i);
end
rssi1 = abs(rssi1);
H1=rssi1(1:3:end,:);
h1=reshape(H1', 1, []);
H1C = repmat(h1,[1,ceil(arr_num/length(h1))]);
H2=rssi1(2:3:end,:);
h2=reshape(H2', 1, []);
H2C = repmat(h2,[1,ceil(arr_num/length(h2))]);
H3=rssi1(3:3:end,:);
h3=reshape(H3', 1, []);
H3C = repmat(h3,[1,ceil(arr_num/length(h3))]);
H =[H1C;H2C;H3C];
H_ave = (mean(H,2))';
% figure
% cdfplot(H1C);
% hold on;
% cdfplot(H2C);
% hold on;
% cdfplot(H3C);
%h=zeros(UEs,arr_num);
P_RB = [];
T_max = [];
Sigma = [];
Flag_p = 1;
rangeP = [0.15:0.01:0.3];
lenP = length(rangeP);
%traffic_types0 = {'poisson','periodic_packet','periodic_batch','bursty'};
%traffic_types = {'periodic_batch','bursty'};
traffic_types = {'poisson'};
err = [0, 0];
th99 = [0, 0];
results = struct();
num = 3000;
for Number1 = 1:num%lenP
    for ty = 1:length(traffic_types)
    rng(Number1);
    Number1
%     p(1)=rangeP(Number1);
%     for Number2 = 1:lenP
%             p(2)=rangeP(Number2);
%         for Number3 = 1:lenP
%             p(3)=rangeP(Number3);
%             p
    p = 0.1+ rand(1,3).*0.2;
    %p = [0.1,0.2,0.3];
	%p = [0.5,0.2,0.1];
    %%0616
    
    T1=zeros(arr_num,UEs);
    T2=zeros(arr_num,UEs);
    T3=zeros(arr_num,UEs);
    T4=zeros(arr_num,UEs);
    queue1=zeros(5,arr_num,UEs);
    d=zeros(UEs,arr_num);

    type = traffic_types{ty};
    fprintf('Running simulation for %s traffic...\n', type);

    % === 仿真数据收集 ===
    all_delays = [];
for j = 1:UEs
    %终端用户类别
    arr_mean(j) = 1/lambda(j);
    %平均到达时间与平均服务时间
    switch type  % 'poisson', 'periodic_packet', 'periodic_batch', 'bursty'
        case 'poisson'
            % -------------------------
            % 经典泊松过程
            % -------------------------
            queue1(1,:,j) = cumsum(exprnd(arr_mean(j),1,arr_num));
            
        case 'periodic_packet'
            % -------------------------
            % 周期性逐包到达（工业控制型）
            % -------------------------
            T = arr_mean(j);
            jitter = 0.00001*T;  % 抖动设置成 1% 的周期，避免偏差过大
            arrivals = T*(1:arr_num) + randn(1,arr_num)*jitter;
            arrivals = max(arrivals,0);   % 去掉负数
            queue1(1,:,j) = sort(arrivals);

        case 'periodic_batch'
            % -------------------------
            % 周期性突发批量到达（AR/VR，传感器聚合）
            % -------------------------
            T_cycle = 0.05;  % 周期长度，按需要调整
            packets_per_cycle = lambda(j) * T_cycle;
            arrivals = [];
            for n = 1:ceil(arr_num/packets_per_cycle)
                n_packets = floor(packets_per_cycle);
    if rand < (packets_per_cycle - n_packets)
        n_packets = n_packets + 1;
    end

    offsets = (0:n_packets-1)*1e-4 + rand(1,n_packets)*1e-5;
    arrivals = [arrivals, (n-1)*T_cycle + offsets];
            end
            queue1(1,:,j) = arrivals(1:arr_num);

        case 'bursty'
            % -------------------------
            % On-Off 突发流量（MMPP近似）
            % -------------------------
            lambda_on = 3*lambda(j);     
            lambda_off = 0.01*lambda(j); 
            Ton_mean = 0.01;  
            Toff_mean = 0.02;  
            q_temp = [];
            offset = 0;
            while length(q_temp) < arr_num
                Ton = exprnd(Ton_mean);
                Toff = exprnd(Toff_mean);
                num_on = ceil(Ton*lambda_on);
                if num_on > 0
                    arr_on = cumsum(exprnd(1/lambda_on,1,num_on));
                    q_temp = [q_temp, arr_on + offset];
                    offset = q_temp(end);
                end
                offset = offset + Toff;
            end
            queue1(1,:,j) = q_temp(1:arr_num);
    end
   
    ser_mean2(j) = Tc(j);
    %按负指数分布产生各顾客达到时间间隔
    %queue1(1,:,j) = exprnd(arr_mean(j),1,arr_num);
    %各顾客的到达时刻等于时间间隔的累积和
    %queue1(1,:,j) = cumsum(queue1(1,:,j));
    %按指数分布随机产生每个任务的数据大小(数据包大小固定)
    %d(j,:)=exprnd(D(j),1,arr_num);
    %按瑞利分布随机产生每个任务发送时的信道增益
    %h(j,:)=raylrnd(sigma,1,arr_num);
    %按负指数分布产生各顾客服务时间
    %queue1(2,:,j) = exprnd(ser_mean1(j),1,arr_num);
    %按香农公式计算瞬时传输速率 利用数据大小计算服务时间
    %queue1(2,:,j)= arrayfun(@(x,y) x/(B(j)*log2(1+y*P(j)/(B(j)*N0) )),d(j,:),h(j,:));

    
    %%new allocation
    n_RB=binornd(N_RB,p(j));
    while n_RB==0
        queue1(2,1,j)=queue1(2,1,j)+t0;
        n_RB=binornd(N_RB,p(j));
    end
    queue1(2,1,j)=queue1(2,1,j)+D(j)/(n_RB*B(j)*log2(1+(H(j,1))*Pow(j)/(B(j)*N0)));
  
    %计算仿真顾客个数，即到达时刻在仿真时间内的顾客数  
    len_sim(j) = sum(queue1(1,:,j)<= Total_time);
    %*****************************************
    %计算第 1个顾客的信息
    %*****************************************
    %第 1个顾客进入系统后直接接受服务，无需等待 
    queue1(3,1,j) = 0;
    %其离开时刻等于其到达时刻与服务时间之和(额外的等待时间)
    queue1(4,1,j) = queue1(1,1,j)+queue1(2,1,j);
    T1(1,j)=queue1(1,1,j);
    T2(1,j)=queue1(2,1,j);
    T3(1,j)=queue1(3,1,j);
    T4(1,j)=queue1(4,1,j);
    time1(1,j)=T4(1,j)-T1(1,j);
    class1(1,j)=j;
    %其肯定被系统接纳，此时系统内共有
    %1个顾客，故标志位置1
    queue1(5,1,j) = 1;
    %其进入系统后，系统内已有成员序号为 1
    member1 = [1];
    for i = 2:arr_num
        %如果第 i个顾客的到达时间超过了仿真时间，则跳出循环
        if queue1(1,i,j)>Total_time 
        break;
        else
            number1 = sum(queue1(4,member1,j) > queue1(1,i,j));
            %如果系统已满，则系统拒绝第 i个顾客，其标志位置 0
            if number1 >= N+1
                queue1(5,i,j) = 0;
                %如果系统为空，则第 i个顾客直接接受服务
            else
                
                %%old allocation
                n_RB=binornd(N_RB,p(j));
                while n_RB==0
                    queue1(2,i,j)=queue1(2,i,j)+t0;
                    n_RB=binornd(N_RB,p(j));
                end
                queue1(2,i,j)=queue1(2,i,j)+D(j)/(n_RB*B(j)*log2(1+(H(j,i))*Pow(j)/(B(j)*N0)));
                
                if number1 == 0
                    
                    %其等待时间为 0
                    %PROGRAMLANGUAGEPROGRAMLANGUAGE
                    queue1(3,i,j) = 0;
                    %其离开时刻等于到达时刻与服务时间之和
                    queue1(4,i,j) = queue1(1,i,j)+queue1(2,i,j);
                    %其标志位置 1
                    queue1(5,i,j) = 1;
                    member1 = [member1,i];
                    %如果系统有顾客正在接受服务，且系统等待队列未满，则 第 i个顾客进入系统
                else
                    len_mem1 = length(member1);
                    
                    
                    %其等待时间等于队列中前一个顾客的离开时刻减去其到 达时刻
                    queue1(3,i,j)=queue1(4,member1(len_mem1),j)-queue1(1,i,j);
                    %其离开时刻等于队列中前一个顾客的离开时刻加上其服务时间
                    queue1(4,i,j)=queue1(4,member1(len_mem1),j)+queue1(2,i,j);
                    %标识位表示其进入系统后，系统内共有的顾客数
                    queue1(5,i,j) = number1+1;
                    member1 = [member1,i];
                end
            end  
        end
        T1(i,j)=queue1(1,i,j);
        T2(i,j)=queue1(2,i,j);
        T3(i,j)=queue1(3,i,j);
        T4(i,j)=queue1(4,i,j);
        time1(i,j)=T4(i,j)-T1(i,j);
        class1(i,j)=j;
    end
end
%仿真结束时，进入系统的总顾客数
%len_mem(j) = length(member1);
%T4为离开时间
%Inter排队等待时间
%T1为到达时间
%%服务器处各时间表示
%终端用户处排队情况queue1
[order,class2]=merge(T4,class1,1,UEs);
queue2(1,:) = order;%到达时间
queue2(2,1) = exprnd(ser_mean2(class2(1)),1);%服务时间
queue2(3,1) = 0;%等待时间
queue2(4,1) = queue2(1,1) + queue2(2,1);%离开时间
queue2(5,1) = 1;%系统中的顾客数目
member2 = [1];%系统中顾客的编号
f(1) = queue2(4,1);
k=2;
%for k = 2:arr_num
while(k<=length(order))
    number2 = sum(queue2(4,member2) > order(k));
    %如果系统已满，则系统拒绝第 i个顾客，其标志位置 0
    if number2 >= N+1
        queue2(5,k) = 0;
        %如果系统为空，则第 i个顾客直接接受服务
    else
        queue2(2,k) = exprnd(ser_mean2(class2(k)),1);
        if number2 == 0
            %其等待时间为 0
            %PROGRAMLANGUAGEPROGRAMLANGUAGE
            queue2(3,k) = 0;
            %其离开时刻等于到达时刻与服务时间之和
            queue2(4,k) = queue2(1,k)+queue2(2,k);
            %其标志位置 1
            queue2(5,k) = 1;
            member2 = [member2,k];
            %如果系统有顾客正在接受服务，且系统等待队列未满，则 第 i个顾客进入系统
        else
            len_mem2 = length(member2);
            %其等待时间等于队列中前一个顾客的离开时刻减去其到 达时刻
            queue2(3,k)=queue2(4,member2(len_mem2))-queue2(1,k);
            %其离开时刻等于队列中前一个顾客的离开时刻加上其服
            %务时间
            queue2(4,k)=queue2(4,member2(len_mem2))+queue2(2,k);
            %标识位表示其进入系统后，系统内共有的顾客数
            queue2(5,k) = number2+1;
            member2 = [member2,k];
        end
    end 
    f(k)=queue2(4,k);
    k=k+1;   
end
%MEC服务器处排队情况queue2 
time2 = queue2(4,:)-queue2(1,:);
%plot(time2)
time = [];
m = 1;
flag = ones(1,UEs);
%队列1和队列2对应相加得总的端到端时延
while(m<=length(time2))
    for n = 1:UEs
       if class2(m) == n
           time(m) = time1(flag(n),n)+time2(m);
           flag(n) = flag(n)+1;
       end
    end
    m = m + 1;
end

% 数据概率密度图以及分布函数图 and 有线信道的理论时延分布
x=[];
m=1;
while(queue2(1,m)<0.1)
    m=m+1;
end
mean_time2 = mean(time2(m:end));
Sigma = [Sigma ; 1-mean_mu/(Fc*mean_time2)];

a1=find(class2==1);
a1(find(a1<m))=[];
x1=time(a1);
Len=length(x1);
a2=find(class2==2);
a2(find(a2<m))=[];
x2=time(a2);
Len=min(Len,length(x2));
a3=find(class2==3);
a3(a3<m)=[];
x3=time(a3);%时延数据样本的概率密度分布
Len=min(Len,length(x3));

x = [x1(1:Len);x2(1:Len);x3(1:Len)];

% EVT tail estimate
esti_para = [];
threshold = [];
for k = 1:UEs
iteration = 200;
epsilon = 0.01;
%m_latencyMax = 0.0002;
max_x = 0;
p1_T = 10;
Out=[];
Out_delta=[];
Out_xi=[];
Mini = [];
for i=1:Len
    for j=i+1:Len
        if x(k,i)<x(k,j)
            max_x = x(k,i);
            x(k,i) = x(k,j);
            x(k,j) = max_x;
        end
    end
end
K = floor(Len/p1_T);
threshold(k) = x(k,K+1);
for i=1:K
    x(k,i)=x(k,i)-threshold(k);
end
tmpt = 1+x(k,1)^2;
xi_old = 1; delta_old = 0.01;step_d = 0.001;step_x = 0.001;
for i=1:200
    delta = delta_old;
    xi = xi_old;
    sum_d = 0;
    sum_x = 0;
    for j=1:K
        sum_d = sum_d + grad_delta(x(k,j),delta,xi);
        sum_x = sum_x + grad_xi(x(k,j),delta,xi);
    end
    mu_d = sum_d/K;
    mu_x = sum_x/K;
    delta_0 = delta;
    xi_0 = xi;
    
    for j=1:iteration
        index = unidrnd(K);
        diff_d = step_d * (grad_delta(x(k,index),delta_0,xi_0)-grad_delta(x(k,index),delta,xi)+ mu_d);
        delta_t = delta_0 - diff_d;
        diff_x = step_x * (grad_xi(x(k,index),delta_0,xi_0)-grad_xi(x(k,index),delta,xi)+ mu_x);
        xi_t = xi_0 - diff_x;
        if delta_t <= 0
            delta_t = delta_0;
            xi_t = xi_0;
            step_d = step_d/2;
            %step_x = step_x/2;	   
        end
        if xi_t<0&&1+xi_t*x(k,1)/delta_t<=0
            xi_t = (xi_t-delta_t*x(k,1))/tmpt;
            delta_t = xi_t*(-x(k,1)) + 0.00001;
            %xi_t = xi_t + 0.001;
        end
%         if xi_t<-1
%             xi_t = -0.5;
%             delta_t = xi_t*(-x(k,1));
%             delta_t = delta_t + 0.00001;
%         end
        delta_0 = delta_t;
        xi_0 = xi_t;
        Out = [Out, min_function(x(k,:),K,delta_t,xi_t)];
        Out_delta = [Out_delta, delta_0];
        Out_xi = [Out_xi, xi_0];
    end
    delta_old = delta_t;
    xi_old = xi_t;
    %[Mini , min_function(x1,K,delta_t,xi_t)];
end
esti_para = [esti_para; delta_old, xi_old];

T_Max(k) = esti_para(k,1)/esti_para(k,2)*((epsilon*Len/K).^(-esti_para(k,2))-1)+threshold(k);
end
P_RB = [P_RB;p];
T_max = [T_max;T_Max];
Data = [Data; p,T_Max];
clearvars queue1 queue2
%         end
%     end
end
end


% theoretical analysis

UE = 3;
%mu = 1*10^5;
mu0 = B(1).*log2(1+H.*Pow(1)./(N0.*B(1)));
mu = (mean(mu0,2))';
mec = 3*10^4;
tau = 0.1*10^(-3);
lam= [2.5,3,3.5]*10^3;
w = lam/sum(lam);
c = N_RB*mu.*p;
%q = N/tau*p;
q = (1-(1-p).^N_RB)/tau;
item1 = c*mec*(1-sigma)/UE+lam.*(c+q);
item2 = q*mec*(1-sigma)/UE+lam.*(c+q);
f = sum(w.*lam.^2.*(c+q).^2./item1./item2)-sigma;
zeros = solve(f==0, sigma);
num = vpa(zeros,4)
num01 = num(num > 0 & num < 0.99)
v = mec*(1-num01)
s1 = (c+q-lam-sqrt((c-q).^2+lam.^2+2.*(c+q).*lam))/2;
s2 = (c+q-lam+sqrt((c-q).^2+lam.^2+2.*(c+q).*lam))/2;
F =  1-s1.*s2./(s1-s2).*(exp(-v.*t).*(1./(v-s1)-1./(v-s2))+v.*exp(-s2.*t)./(s2.*(v-s2))-v.*exp(-s1.*t)./(s1.*(v-s1)));
F1 = vpa(F(1),4);
F2 = vpa(F(2),4);
F3 = vpa(F(3),4);
double(subs(F1,t,298));
aver1 = int(diff(F1,t)*t,t,0,inf);
aver2 = int(diff(F2,t)*t,t,0,inf);
aver3 = int(diff(F3,t)*t,t,0,inf);
per1 = double(vpa(vpasolve(F1==0.99,[0,1]),8))
per2 = double(vpa(vpasolve(F2==0.99,[0,1]),8))
per3 = double(vpa(vpasolve(F3==0.99,[0,1]),8))
