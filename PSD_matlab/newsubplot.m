function ax = newsubplot(position, xlab, ylab, ttl)
    % Creates new subplot in specified position on current figure
    % with xlab xlabel and ylab ylabel
    ax = subplot(position); 
    hold on
    set(ax,'FontSize',14, 'FontName','Times New Roman') %and other properties
    title(ttl, 'FontSize', 18, 'FontWeight', 'bold', 'FontName','Times New Roman')
    xlabel(xlab); ylabel(ylab)
    grid on
end