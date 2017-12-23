function []=solveProblem(no_of_points_per_class)
  if(nargin<1)
    no_of_points_per_class=1000;
  end

  dbmoon(no_of_points_per_class);
  load dbmoon;
  
%  A naive O(n^2) algorithm is used to 
%  find the max distance between points of dataset 
%  to initialize the standard deviation of the RBF nodes  

  maximum_dis=0;
  for p1=1:(-1+2*no_of_points_per_class)
%  Extract the points from p1+1 to end
    points=data(p1+1:end,1:2);
    
    delta_p_p1=(points-repmat(data(p1,1:2),size(points,1),1))';
    cur_maximal_dis=max(sum((delta_p_p1.*delta_p_p1)));
    if(cur_maximal_dis>maximum_dis)
      maximum_dis=cur_maximal_dis;
    end
  end

   disp( sprintf('Max dis is %f\n',maximum_dis))
  
  for no_of_means=1:8
    means=applyKMeans(no_of_means,data);
    disp( sprintf('\nNo of means is %d\n',int16(no_of_means)));
    tic;
    eta=0.05;
    W=((rand(no_of_means+1,1))-1)*0.5; % W0 is the bias term
    sigma=maximum_dis/sqrt(2*2*no_of_points_per_class);
    for it=1:250
      for po=1:round(1.5*no_of_points_per_class)
        pattern=round(1.99*no_of_points_per_class*rand())+1;
        point=data(pattern,1:2);
    %   Forward Path
        delta=means-repmat(point,no_of_means,1);
        net=sum(((delta).*(delta))')';
        act=exp(-(net.*net)/(2*sigma*sigma));
        o=(W')*[1;act];
        t=data(pattern,3);
    %   Backpropagation
        W=W+(eta*(t-o)*([1;act]));
        for d=1:2
            means(:,d)=means(:,d)+(eta*(t-o)*(W(2:end).*act).*delta(:,d))/(sigma*sigma);
        end
        sigma=sigma+(1/(sigma*sigma*sigma))*(t-o)*sum(W(2:end).*act.*net);
      end
    end
    toc;
    
    sum_of_squared_error=0;  
    classification_error=0;
    for pattern=1:2000
      point=data(pattern,1:2);
    %   Forward Path
        net=sum(((means-repmat(point,no_of_means,1)).*(means-repmat(point,no_of_means,1)))');
        act=exp(-(net.*net)/(2*sigma*sigma));
        o=(W')*[1;act'];
        t=data(pattern,3);
        sum_of_squared_error=sum_of_squared_error+((t-o)*(t-o));
        if(o>0.5)
          o=1;
        else
          o=0;
        end
        classification_error=classification_error+abs(t-o);
    end
    disp( sprintf('Classeification error is %d from %d patterns\n',classification_error,2*no_of_points_per_class))
    disp( sprintf('Total Squared Error is %f\n',sum_of_squared_error))
    
    subplot(2,4,no_of_means)
    plot(data(1:N,1),data(1:N,2),'.r',data(N+1:end,1),data(N+1:end,2),'.b',means(:,1),means(:,2),'.g')
  end
  disp( sprintf('A hidden layer with 6 nodes is enough to achieve the required accuracy'))
  

function data=dbmoon(N,d,r,w)
  % Usage: data=dbmoon(N,d,r,w)
  % doublemoon.m - genereate the double moon data set in Haykin's book titled
  % "neural networks and learning machine" third edition 2009 Pearson
  % Figure 1.8 pp. 61
  % The data set contains two regions A and B representing 2 classes
  % each region is a half ring with radius r = 10, width = 6, one is upper
  % half and the other is lower half
  % d: distance between the two regions
  % will generate region A centered at (0, 0) and region B is a mirror image
  % of region A (w.r.t. x axis) with a (r, d) shift of origin
  % N: # of samples each class, default = 1000
  % d: seperation of two class, negative value means overlapping (default=1)
  % r: radius (default=10), w: width of ring (default=6)
  % 
  % (C) 2010 by Yu Hen Hu
  % Created: Sept. 3, 2010

  % clear all; close all;
  if (nargin<4) 
    w=6;
  end
  if (nargin<3) 
    r=10;
  end
  if (nargin<2) 
    d=1;
  end
  if (nargin < 1) 
    N=1000;
  end

  % generate region A:
  % first generate a uniformly random distributed data points from (-r-w/2, 0)
  % to (r+w/2, r+w/2)
  N1=10*N;  % generate more points and select those meet criteria
  w2=w/2; 
  done=0; data=[]; tmp1=[];
  while ~done, 
      tmp=[2*(r+w2)*(rand(N1,1)-0.5) (r+w2)*rand(N1,1)];
      % 3rd column of tmp is the magnitude of each data point
      tmp(:,3)=sqrt(tmp(:,1).*tmp(:,1)+tmp(:,2).*tmp(:,2)); 
      idx=find([tmp(:,3)>r-w2] & [tmp(:,3)<r+w2]);
      tmp1=[tmp1;tmp(idx,1:2)];
      if length(idx)>= N, 
          done=1;
      end
      % if not enough data point, generate more and test
  end
  % region A data and class label 0
  % region B data is region A data flip y coordinate - d, and x coordinate +r
  data=[tmp1(1:N,:) zeros(N,1);
      [tmp1(1:N,1)+r -tmp1(1:N,2)-d ones(N,1)]];

  plot(data(1:N,1),data(1:N,2),'.r',data(N+1:end,1),data(N+1:end,2),'.b');
  axis([-r-w2 2*r+w2 -r-w2-d r+w2])
  title(['Fig. 1.8 Double moon data set, d = ' num2str(d)]),

  save dbmoon N r w d data;

function [means]=applyKMeans(noOfMeans,data)
  means=rand(noOfMeans,2);
  N=size(data,1);
  nearestMean=zeros(1,N);
  delta=zeros(noOfMeans,2);
  dis=zeros(noOfMeans,1);
  for it=1:250
    %Assign each pattern to its the nearest mean  
    for pattern=1:N
      delta=means-repmat(data(pattern,1:2),size(means,1),1);
      delta=delta.*delta;
      dis=delta*ones(2,1);
      [~,nearestMean(pattern)]=min(dis); 
    end
    
    %Compute the new means
    for meanIn=1:noOfMeans
      count=sum(nearestMean==meanIn);
      if(count>0)
        means(meanIn,:)=((nearestMean==meanIn)*(data(:,1:2)))/count;  
      end
    end
  end 
