COM_x_p = sub.outPut(:,40);
COM_y_p = sub.outPut(:,41);
COM_x_a = sub.outPut(:,43);

pos.toes_r.x  = sub.outPut(:,12)+0.08;
pos.toes_r.y  = sub.outPut(:,13);
pos.calcn_r.x = sub.outPut(:,14);
pos.calcn_r.y = sub.outPut(:,15);
pos.toes_l.x  = sub.outPut(:,16)+0.08;
pos.toes_l.y  = sub.outPut(:,17);
pos.calcn_l.x = sub.outPut(:,18);
pos.calcn_l.y = sub.outPut(:,19);
zmp_x = COM_x_p-COM_x_a.*COM_y_p./9.81; 

low = min(pos.calcn_r.x,pos.calcn_l.x);
high = max(pos.toes_r.x,pos.toes_l.x);

subplot(2,1,1)
plot( [low-zmp_x, zmp_x-zmp_x, high-zmp_x])
subplot(2,1,2)
plot( [zmp_x, low, high])