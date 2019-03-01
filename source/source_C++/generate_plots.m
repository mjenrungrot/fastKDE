fileloc = '../Exp/norm2d';
ofileloc = '../Exp/out_norm2d';

deltas = 10.^[-12:-1];
D1 = repmat(deltas',1,12);
D2 = repmat(deltas,12,1);

BWS = {'.625','.2500'};
OBWS = {'-625','-2500'};
%BWS = {'.625','.2500'};

for i=1:length(BWS)
    A = load([fileloc BWS{i} '.prune.time.txt']);
    fig=figure;
    h=surf(D1,D2,A);
    set(get(h,'Parent'),'XScale','log');
    set(get(h,'Parent'),'YScale','log');
    xlim([10^-12 1]);
    ylim([10^-12 1]);
    zlim([0 max(max(A(:)),.02)]);
    t=xlabel('\delta_{exclude}', 'FontSize', 14);
    set(t, 'FontSize', 14);
    t=ylabel('\delta_{mean}');
    set(t, 'FontSize', 14);
    t=zlabel('Mean Runtime Per Query (s)');
    set(t, 'FontSize', 14);
    view([137 38])
    t=title('Term-Based Algorithm Runtime vs. Deltas');
    set(t, 'FontSize', 14);
    set(get(h,'Parent'),'FontSize', 14);
    saveas(fig,[ofileloc OBWS{i} '-prune-time.fig'])
    saveas(fig,[ofileloc OBWS{i} '-prune-time.eps'],'psc2')
    term_times = A(:);

    A = load([fileloc BWS{i} '.prune.accu.txt']);
    fig=figure;
    h=surf(D1,D2,A);
    set(get(h,'Parent'),'XScale','log');
    set(get(h,'Parent'),'YScale','log');
    xlim([10^-12 1]);
    ylim([10^-12 1]);
    t=xlabel('\delta_{exclude}');
    set(t, 'FontSize', 14);
    t=ylabel('\delta_{mean}');
    set(t, 'FontSize', 14);
    t=zlabel('% Error');
    set(t, 'FontSize', 14);
    t=title('Term-Based Algorithm Accuracy vs. Delta');
    set(t, 'FontSize', 14);
    set(get(h,'Parent'),'FontSize', 14);
    saveas(fig,[ofileloc OBWS{i} '-prune-accu.fig'])
    saveas(fig,[ofileloc OBWS{i} '-prune-accu.eps'],'psc2')
    term_accus = A(:);

    A = load([fileloc BWS{i} '.int.comp.time.txt']);
    B = load([fileloc BWS{i} '.int.dist.time.txt']);
    fig=figure;
    h=plot(deltas,A,'LineWidth',2);
    set(get(h,'Parent'),'XScale','log');
    xlim([10^-12 1]);
    ylim([0 max(max(max(A(:)),max(B(:))),.02)]);
    hold on;
    plot(deltas,B,'--r','LineWidth',2);
    t=legend('EPP Priority', 'Distance Priority');
    set(t, 'FontSize', 14);
    t=xlabel('\delta');
    set(t, 'FontSize', 14);
    t=ylabel('Mean Runtime Per Query (s)');
    set(t, 'FontSize', 14);
    t=title('Interval-Based Algorithm Runtime vs. Deltas');
    set(t, 'FontSize', 14);
    set(get(h,'Parent'),'FontSize', 14);
    saveas(fig,[ofileloc OBWS{i} '-int-time.fig'])
    saveas(fig,[ofileloc OBWS{i} '-int-time.eps'],'psc2')
    int_cmp_times = A(:);
    int_dis_times = B(:);

    A = load([fileloc BWS{i} '.int.comp.accu.txt']);
    B = load([fileloc BWS{i} '.int.dist.accu.txt']);
    fig=figure;
    h=plot(deltas,A,'LineWidth',2);
    set(get(h,'Parent'),'XScale','log');
    xlim([10^-12 1]);
    hold on;
    plot(deltas,B,'--r','LineWidth',2);
    t=legend('EPP Priority', 'Distance Priority','Location','Best');
    set(t, 'FontSize', 14);
    t=xlabel('\delta');
    set(t, 'FontSize', 14);
    t=ylabel('% Error');
    set(t, 'FontSize', 14);
    t=title('Interval-Based Algorithm Accuracy vs. Delta');
    set(t, 'FontSize', 14);
    set(get(h,'Parent'),'FontSize', 14);
    saveas(fig,[ofileloc OBWS{i} '-int-accu.fig'])
    saveas(fig,[ofileloc OBWS{i} '-int-accu.eps'],'psc2')
    int_cmp_accus = A(:);
    int_dis_accus = B(:);

    fig=figure;
    h=scatter(term_times,term_accus,100);
    hold on
    scatter(int_cmp_times,int_cmp_accus,100,'+');
    scatter(int_dis_times,int_dis_accus,100,'x');
    t=legend('Term-Based', 'Interval EPP', 'Interval Dist');
    set(t, 'FontSize', 14);
    t=xlabel('Mean Runtime Per Query (s)');
    set(t, 'FontSize', 14);
    t=ylabel('% Error');
    set(t, 'FontSize', 14);
    t=title('Accuracy vs. Runtime');
    set(t, 'FontSize', 14);
    set(get(h,'Parent'),'FontSize', 14);
    saveas(fig,[ofileloc OBWS{i} '-accuvrun.fig'])
    saveas(fig,[ofileloc OBWS{i} '-accuvrun.eps'],'psc2')
end