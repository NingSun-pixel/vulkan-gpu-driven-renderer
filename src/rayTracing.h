#include <glm/gtx/quaternion.hpp>
#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

//Demo 0: CPU-Mutithread Ray tracing
class Ray {
public:
	glm::vec3 dir;
	glm::vec3 origin;
	Ray(glm::vec3 o, glm::vec3 d) : origin(o), dir(d) {}
	glm::vec3 at(float t) const;


private:
};


class hit_record {
public:
	glm::vec3 p;
	glm::vec3 normal;
	double t;
	bool front_face;

	void set_face_normal(const Ray& r, const glm::vec3& outward_normal);

};





class hittable {
public:
	virtual ~hittable() = default;
	virtual bool hit(const Ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const = 0;

};

class sphere : public hittable {
public:
	sphere(const glm::vec3& center, double radius) : center(center), radius(std::fmax(0, radius)) {};

	bool hit(const Ray& r, double ray_tmin, double ray_tmax, hit_record& rec)const override;


private:
    glm::vec3 center;
    double radius;
};


class hittable_list : public hittable {
public:
    std::vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
    }

	bool hit(const Ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override;
};
