
import matplotlib.pyplot as plt

import seaborn as sns

def salon(ultramerge, split, PATH, MATERIA=None):
# they went more than 3, and they passed
    group1 = ultramerge[(ultramerge['VISITAS'] > split) & ((ultramerge['CALIFICACION'].notna()) & (ultramerge['CALIFICACION'] >= 7.5))]

    # they went more than 3, and they did not pass
    group2 = ultramerge[(ultramerge['VISITAS'] > split) & ((ultramerge['CALIFICACION'].isna()) | (ultramerge['CALIFICACION'] < 7.5))]

    # they went less than 3, and they passed
    group3 = ultramerge[(ultramerge['VISITAS'] <= split) & ((ultramerge['CALIFICACION'].notna()) & (ultramerge['CALIFICACION'] >= 7.5))]

    # they went less than 3, and they did not pass
    group4 = ultramerge[(ultramerge['VISITAS'] <= split) & ((ultramerge['CALIFICACION'].isna()) | (ultramerge['CALIFICACION'] < 7.5))]

    # get the percentage of each group
    total_students = len(ultramerge)
    perc_group1 = (len(group1) / total_students) * 100
    perc_group2 = (len(group2) / total_students) * 100
    perc_group3 = (len(group3) / total_students) * 100
    perc_group4 = (len(group4) / total_students) * 100


    if split == 1:
        keyword = 'visita'
    else:
        keyword = 'visitas'
    # SCATTERPLOT, x=VISITAS, y=IMPKDE_Z
    # remove the border of each point with sns.scatterplot
    sns.scatterplot(data=group1, x='VISITAS', y='IMPKDE_Z', alpha=0.5, label=f'Más de {split} {keyword} (pasaron) - {perc_group1:.2f}%', edgecolor=None)
    sns.scatterplot(data=group2, x='VISITAS', y='IMPKDE_Z', alpha=0.5, label=f'Más de {split} {keyword} (no pasaron) - {perc_group2:.2f}%', edgecolor=None)
    sns.scatterplot(data=group3, x='VISITAS', y='IMPKDE_Z', alpha=0.5, label=f'Menos de {split} {keyword} (pasaron) - {perc_group3:.2f}%', edgecolor=None)
    sns.scatterplot(data=group4, x='VISITAS', y='IMPKDE_Z', alpha=0.5, label=f'Menos de {split} {keyword} (no pasaron) - {perc_group4:.2f}%', edgecolor=None)
    plt.title('Relación entre Número de visitas y Calificación por Salón')
    plt.xlabel('Número de Visitas al CMAT')
    plt.ylabel('Calificación Estandarizada (KDE, Z-score)')
    plt.legend(loc='lower right')

    if MATERIA is not None:
        plt.suptitle(f'Materia: {MATERIA}', y=1.02, fontsize=16)

    plt.savefig(PATH + '07_00.pdf', bbox_inches='tight')

    plt.show()


def estudiante_ultramerge_means(ultramerge_means, split, PATH):
    # SCATTERPLOT, x=VISITAS, y=IMPKDE_Z
    # remove the border of each point with sns.scatterplot
    keyword = 'visitas' if split != 1 else 'visita'
    sns.scatterplot(data=ultramerge_means[ultramerge_means['VISITAS'] > split], x='VISITAS', y='MEAN_IMPKDE_Z', alpha=0.5, label=f'Más de {split} {keyword}', edgecolor=None)
    sns.scatterplot(data=ultramerge_means[ultramerge_means['VISITAS'] <= split], x='VISITAS', y='MEAN_IMPKDE_Z', alpha=0.5, label=f'Menos de {split} {keyword}', edgecolor=None)


    plt.title('Relación entre Número de visitas y Calificación por Estudiante')
    plt.xlabel('Número de Visitas')
    plt.ylabel('Media de Calificaciones Estandarizadas (KDE, Z-score)')
    plt.legend(loc='lower right')

    plt.savefig(PATH + '08_00.pdf', bbox_inches='tight')

    plt.show()