import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# compare the zscore of each group and test the difference of means
def pruebas_parametricas(group1, group2, split, estudiante_o_calificacion='Salón', PATH=None, MATERIA=None):
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    if p_value < 0.05:
        print(f"La diferencia es estadísticamente significativa (p = {p_value:.4f})")
    else:
        print(f"La diferencia no es estadísticamente significativa (p = {p_value:.4f})")
   
   
    if split == 1:
        keyword = 'visita'
    else:
        keyword = 'visitas'

    # verify if group1 is distributed normally
    stat, p = stats.shapiro(group1)
    print()
    if p > 0.05:
        print(f"El Grupo 1 se distribuye normalmente")
    else:
        print("El Grupo 1 no se distribuye normalmente")
    # qqplot 
    sm.qqplot(group1, line ='s')
    plt.title(f'Q-Q Plot para Grupo 1 (Más de {split} {keyword})')
    # change the xlabel and ylabel
    plt.xlabel('Cuantiles Teóricos')
    plt.ylabel('Cuantiles Muestrales')
    if MATERIA is not None:
        plt.suptitle(f'Materia: {MATERIA}', y=1.02, fontsize=16)

    plt.savefig(f'{PATH}08_Parametricas_{estudiante_o_calificacion}_NoNormalidad.pdf', bbox_inches='tight')
    plt.show()
   
    text = f"""
    T-estadístico: {t_stat:.2f}, P-valor: {p_value:.4f} 
    Media de más de {split} visitas: {group1.mean():.2f} 
    Media de {split} o menos visitas: {group2.mean():.2f} 
    Diferencia de medias: {group1.mean() - group2.mean():.2f} 
    Tamaño de muestra más de {split} visitas: {len(group1)} 
    Tamaño de muestra {split} o menos visitas: {len(group2)}
    Shapiro-Wilk Para Grupo 1 (Más de {split} visitas) - Estadistico: {stat}, P-valor: {p}
    """
    print(text.format(
        t_stat=t_stat,
        p_value=p_value,
        group1=group1,
        group2=group2,
        len=len
    ))

    # get the percentages
    total = len(group1) + len(group2)
    perc1 = (len(group1) / total) * 100 
    perc2 = (len(group2) / total) * 100

    sns.kdeplot(group1, cut=0, bw_adjust=0.5, fill=True, alpha=0.3, label=f'Más de {split} {keyword} ({perc1:.2f}%)')
    sns.kdeplot(group2, cut=0, bw_adjust=0.5, fill=True, alpha=0.3, label=f'Menos de {split} {keyword} ({perc2:.2f}%)')
    plt.title(f'Comparación de Distribuciones de Calificaciones por {estudiante_o_calificacion}')
    plt.xlabel('Calificación Estandarizada (KDE, Z-score)')
    plt.ylabel('Densidad de Calificaciones')
    if estudiante_o_calificacion == 'Estudiante':
        plt.ylabel('Densidad de la Media de Calificaciones')
    
    plt.vlines(x=group1.mean(), ymin=0, ymax=1.2, colors='blue', linestyles='-', label=f'Media de más de {split} {keyword}: {group1.mean():.2f}', alpha=0.7)
    plt.vlines(x=group2.mean(), ymin=0, ymax=1.2, colors='orange', linestyles='-', label=f'Media de menos de {split} {keyword}: {group2.mean():.2f}', alpha=0.7)
    plt.legend(loc='upper left')
    plt.xlim(-4, 2)
    if MATERIA is not None:
        plt.suptitle(f'Materia: {MATERIA}', y=1.02, fontsize=16)
    plt.savefig(f'{PATH}08_Parametricas_{estudiante_o_calificacion}.pdf', bbox_inches='tight')
    plt.show()